#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_gl_interop.h>
#include <curand.h>

#include <raylib.h>

#define ceil_div(a, b) (((a) + ((b) - 1)) / (b))

// --- Window & camera settings ---

constexpr int INITIAL_WINDOW_WIDTH = 1280;
constexpr int INITIAL_WINDOW_HEIGHT = 720;

constexpr float MIN_ZOOM = 0.05f;
constexpr float MAX_ZOOM = 16.0f;
constexpr float ZOOM_STRENGTH = 1.1f;

constexpr char APPLICATION_NAME[] = "CUDA + Raylib Rigidbody Simulation";

// --- Simulation settings ---

constexpr float SIM_WIDTH = 13600.0f;
constexpr float SIM_HEIGHT = 7650.0f;

constexpr unsigned OBJECT_COUNT = 1'000'000;
constexpr float OBJECT_RADIUS = 2.0f;
constexpr float DOUBLE_RADIUS = 2.0f * OBJECT_RADIUS;

constexpr float TIME_STEP = 1.0f / 60.0f;
constexpr float MAX_FRAME_TIME = 2.0f * TIME_STEP;

constexpr float GRAVITY = 40.0f;

// Minimum speed required before bouncing is applied.
constexpr float BOUNCE_SPEED_THRESHOLD = 60.0f;
constexpr float BOUNCINESS = 0.1f;

// Apply a small amount of damping constantly, otherwise objects will never
// settle.
constexpr float DAMPING_PER_STEP = 0.999f;

// Terminal velocity.
constexpr float MAX_SPEED = 900.0f;

// Prevents explosions when objects overlap deeply.
constexpr float MAX_SEPARATION_PER_STEP = 0.2f;

// Controls how intense the separation force is for overlapping objects.
constexpr float SEPARATION_SPEED = 0.15f;
constexpr float SEPARATION_TOLERANCE = 0.02f;

// Prevents explosions when objects overlap deeply.
constexpr float MAX_SEPARATION_SPEED = 25.0f;

constexpr float FRICTION = 0.35f;

// Higher values make collisions stop objects more harshly.
constexpr float COLLISION_SHARE = 1.0f;

// Partition the simulation world into a grid to allow for efficient nearby
// object lookup.
constexpr float CELL_SIZE = 8.0f;
constexpr int GRID_WIDTH = (SIM_WIDTH + CELL_SIZE - 1.0f) / CELL_SIZE;
constexpr int GRID_HEIGHT = (SIM_HEIGHT + CELL_SIZE - 1.0f) / CELL_SIZE;
constexpr int CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;
constexpr int MAX_CELL_CAPACITY = 4 * (int)OBJECT_COUNT;

// Launch argumment for kernels.
constexpr unsigned THREADS_PER_BLOCK = 256;

// Offset x, offset y.
constexpr unsigned QUASIRANDOM_DIMENSIONS = 2;

// --- Data structures ---

// Represents the state of the application's 2D camera, which allows the user to
// pan and zoom.
struct SimCamera {
  float x, y, zoom;
};

// Represents one physical object in the simulation.
struct SimObject {
  float2 position;
  float2 velocity;
  uchar4 color;
};

// Represents the texture that CUDA draws the simulation into.
struct SimTexture {
  Texture2D texture;
  cudaGraphicsResource_t resource;
  int width, height;
};

// Represents the state of the simulation. Buffers allocated on the GPU.
struct SimState {
  // Double buffer to store object states across simulation steps.
  SimObject *objects_a;
  SimObject *objects_b;

  // CUDA kernel launch sizes, for both kernels that operate on objects and
  // kernels that operate on object grid cells.
  unsigned object_block_count;
  unsigned cell_block_count;

  // Random sobol offset for initial object placement.
  unsigned long long random_offset;

  // The following fields store an object grid. The objects are partitioned into
  // spatial grid cells, so that physics calculations can efficiently look up
  // nearby objects for collisions.

  // Stores the number of objects overlapping each cell.
  int *cell_object_counts;
  // Item i is the index into cell_object_idxs which marks the beginning of cell
  // i's slice.
  int *cell_slice_starts;
  // Item i is the index in cell_object_idxs where the next object index is
  // written for cell i.
  int *cell_next_slot;
  // Stores the object indices contained in each cell; this is the actual data
  // of which objects are overlapping which cells.
  int *cell_object_idxs;
  // Scratch space used for parallel summing.
  int *block_sums;
};

// --- Device utility functions ---

// Clamps a value between a min and max value. std::clamp can't be called in
// device code, so we implement it ourselves.
__device__ float clampf(const float value, const float min, const float max) {
  return fminf(fmaxf(value, min), max);
}

__device__ int clampi(const int value, const int minv, const int maxv) {
  return min(max(value, minv), maxv);
}

// Prevent out of bounds index of a buffer.
__device__ int clamp_index(const int index, const int count) {
  return clampi(index, 0, count - 1);
}

// Dot product of a vector.
__device__ float dot2(const float2 a, const float2 b) {
  return a.x * b.x + a.y * b.y;
}

// Scale a vector by a scalar factor.
__device__ float2 scale2(const float2 v, const float factor) {
  return make_float2(v.x * factor, v.y * factor);
}

// Magnitude (length) of a vector.
__device__ float magnitude2(const float2 v) { return sqrtf(dot2(v, v)); }

// Determines whether two circles overlap, and computes the separation vector
// and overlap depth.
__device__ bool circles_overlap(const float2 position_a,
                                const float2 position_b, float2 &separation,
                                float &depth) {
  const float dx = position_b.x - position_a.x;
  const float dy = position_b.y - position_a.y;

  const float distance_squared = dx * dx + dy * dy;
  if (distance_squared >= DOUBLE_RADIUS * DOUBLE_RADIUS)
    return false;

  const float distance = sqrtf(fmaxf(distance_squared, 1.0e-8f));

  separation = make_float2(dx / distance, dy / distance);
  depth = DOUBLE_RADIUS - distance;

  return true;
}

// --- Object grid device functions ---

// Computes which cell a given world position falls in.
__device__ int cell_at_position(const float2 position) {
  const int cell_x = position.x * (1.0f / CELL_SIZE);
  const int cell_y = position.y * (1.0f / CELL_SIZE);
  return clamp_index(cell_y, GRID_HEIGHT) * GRID_WIDTH +
         clamp_index(cell_x, GRID_WIDTH);
}

// Computes a bounding box representing the range of possible cells the object
// could be overlapping. The output coordinates are cell positions in the grid.
__device__ void object_overlapping_cells(const float2 position,
                                         int2 &first_cell, int2 &last_cell) {
  first_cell.x = clamp_index((position.x - OBJECT_RADIUS) * (1.0f / CELL_SIZE),
                             GRID_WIDTH);
  last_cell.x = clamp_index((position.x + OBJECT_RADIUS) * (1.0f / CELL_SIZE),
                            GRID_WIDTH);
  first_cell.y = clamp_index((position.y - OBJECT_RADIUS) * (1.0f / CELL_SIZE),
                             GRID_HEIGHT);
  last_cell.y = clamp_index((position.y + OBJECT_RADIUS) * (1.0f / CELL_SIZE),
                            GRID_HEIGHT);
}

// Computes the first grid cell shared by both objects. This is used to select
// one cell per pair of objects, instead of having to loop through all
// overlapping cells.
__device__ int first_shared_cell(const float2 position_a,
                                 const float2 position_b) {
  int2 object_first, object_last;
  object_overlapping_cells(position_a, object_first, object_last);
  int2 other_first, other_last;
  object_overlapping_cells(position_b, other_first, other_last);

  const int first_col = max(object_first.x, other_first.x);
  const int first_row = max(object_first.y, other_first.y);

  return first_row * GRID_WIDTH + first_col;
}

// --- Object grid kernels ---

// Zeroes out the cell object counts. Called once per cell.
__global__ void zero_cell_object_counts(int *const cell_object_counts) {
  const int cell_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell_idx >= CELL_COUNT)
    return;

  cell_object_counts[cell_idx] = 0;
}

// Populates the object grid cells with the number of objects they overlap.
// Called once per object.
__global__ void compute_cell_object_counts(const SimObject *const objects,
                                           const unsigned object_count,
                                           int *const cell_object_counts) {
  const unsigned object_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (object_idx >= object_count)
    return;

  // For each object, we add one to its overlapping cells' object counts.
  int2 first_cell, last_cell;
  object_overlapping_cells(objects[object_idx].position, first_cell, last_cell);

  for (int row = first_cell.y; row <= last_cell.y; row++)
    for (int col = first_cell.x; col <= last_cell.x; col++)
      // Use atomicAdd to prevent data races.
      atomicAdd(&cell_object_counts[row * GRID_WIDTH + col], 1);
}

// Computes the indices of each grid cell's slice in the object indices array.
// In other words, for each cell, it computes how many objects came before it,
// since that determines how far into the object indices array this cell's data
// starts. For example, if the first three cells have 3, 5, and 2 objects in
// them, then their slices in the object indices array are 0, 3, and 8,
// respectively. The fourth cell's data would then start at index 10. This
// kernel operates at the block level, so these results have to be later
// combined for the whole grid.
__global__ void
compute_blockwise_cell_slice_starts(const int *const cell_object_counts,
                                    int *const cell_slice_starts,
                                    int *const block_sums) {
  // Since we are doing a parallel sum, using shared memory here makes sense.
  __shared__ int block_shared[THREADS_PER_BLOCK];

  // Compute the cell index for this thread.
  const int t_idx = threadIdx.x;
  const int cell_idx = blockIdx.x * THREADS_PER_BLOCK + t_idx;

  // At this point we might be able to return if cell_idx >= CELL_COUNT, but
  // looking at the documentation it seems like this is not entirely safe to do
  // because we use barriers later, which might behave weirdly if some threads
  // return early. So, we just keep all threads.

  // Each thread copies its cell's object count into the block shared memory.
  const int cell_object_count =
      cell_idx < CELL_COUNT ? cell_object_counts[cell_idx] : 0;
  block_shared[t_idx] = cell_object_count;

  // This barrier is necessary because we read from block_shared before the
  // first barrier in the loop.
  __syncthreads();

  // We use a Hillis-Steele scan to compute a running sum of the object counts.
  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
  // Each loop iteration, we halve the number of threads used, combining the
  // sums in log(n) time.
  for (int offset = 1; offset < (int)THREADS_PER_BLOCK; offset *= 2) {
    const int add = t_idx >= offset ? block_shared[t_idx - offset] : 0;
    // We must synchronize before and after the add because some threads might
    // be ahead of or behind the current thread.
    __syncthreads();
    block_shared[t_idx] += add;
    __syncthreads();
  }

  // At this point, each element of the shared memory contains the running sum
  // of everything before it, including itself. However we don't want the value
  // to include itself, we just want the total before it. So for each entry, we
  // subtract its own object count.
  if (cell_idx < CELL_COUNT)
    cell_slice_starts[cell_idx] = block_shared[t_idx] - cell_object_count;

  // The final total for this block is stored at the end of the shared memory.
  // We need the final total of the block for a later step, so we save it here.
  if (t_idx == (int)THREADS_PER_BLOCK - 1)
    block_sums[blockIdx.x] = block_shared[t_idx];
}

// Takes the results of compute_blockwise_cell_slice_starts and combines them to
// form the final data for the entire grid. The result is that each cell knows
// which index into the object indices array it should store the objects it
// contains. See the comment above for an example. This kernel is called with
// only 1 block because it only has 256 values to scan.
__global__ void compute_cell_slice_starts(int *const block_sums,
                                          const unsigned cell_block_count,
                                          int *const cell_slice_starts) {
  // We are doing another parallel sum, so we use shared memory again.
  __shared__ int block_shared[THREADS_PER_BLOCK];

  const int t_idx = threadIdx.x;
  // Evenly divvy up the work across threads. Each thread handles the same
  // number of block totals computed in the previous step.
  const int blocks_per_thread = ceil_div(cell_block_count, THREADS_PER_BLOCK);
  const int start_block_idx = t_idx * blocks_per_thread;
  const int end_block_idx =
      min(start_block_idx + blocks_per_thread, cell_block_count);

  // Sum up the block totals this thread is handling.
  int sum = 0;
  for (int b = start_block_idx; b < end_block_idx; b++)
    sum += block_sums[b];
  block_shared[t_idx] = sum;
  __syncthreads();

  // We use the same Hillis-Steele algorithm as above to compute a running sum.
  for (int offset = 1; offset < (int)THREADS_PER_BLOCK; offset *= 2) {
    const int add = t_idx >= offset ? block_shared[t_idx - offset] : 0;
    __syncthreads();
    block_shared[t_idx] += add;
    __syncthreads();
  }

  // At this point, `sum` holds the total number of objects in the blocks
  // handled by this thread, and `block_shared[t_idx]` holds the running sum of
  // the block totals up to t_idx, including itself.

  // Like before, we want an exclusive sum rather than an inclusive sum, so we
  // subtract our own total from the running total. This is because the
  // exclusive sum is the slice offset.
  int exlusive_sum = block_shared[t_idx] - sum;

  // We reuse the block_sums memory. Now block_sums[i] represents the index into
  // the object indices array where block i's entries start, which is the
  // exlusive sum. We save these values for the next step where we actually
  // populate the cell_slice_starts array.
  for (int block_idx = start_block_idx; block_idx < end_block_idx;
       block_idx++) {
    const int block_sum = block_sums[block_idx];
    block_sums[block_idx] = exlusive_sum;

    // The next entry includes our sum, but not its own, so it stays as an
    // exclusive sum on the next iteration.
    exlusive_sum += block_sum;
  }

  // The cell slice starts array actually has CELL_COUNT + 1 elements because
  // the last element contains the sum of the previous. In other words, the last
  // element is a sentinel which contains the total number of objects in the
  // grid.
  if (t_idx == (int)THREADS_PER_BLOCK - 1) {
    const int total = block_shared[THREADS_PER_BLOCK - 1];
    cell_slice_starts[CELL_COUNT] = min(total, MAX_CELL_CAPACITY);
  }
}

// This kernel is called once per cell and uses the results calculated by the
// previous kernels to populate both the cell_slice_starts and cell_next_slot
// arrays in the simulation state.
__global__ void populate_cell_slice_starts(int *const cell_slice_starts,
                                           int *const cell_next_slot,
                                           const int *const block_indices) {
  const int cell_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (cell_idx >= CELL_COUNT)
    return;

  const int cell_start =
      cell_slice_starts[cell_idx] + block_indices[blockIdx.x];

  cell_slice_starts[cell_idx] = cell_start;
  cell_next_slot[cell_idx] = cell_start;
}

// Populates the object grid cells with object indices, indicating which objects
// are in the cells.
__global__ void fill_cell_object_idxs(const SimObject *const objects,
                                      const unsigned object_count,
                                      const int *const cell_slice_starts,
                                      int *const cell_next_slot,
                                      int *const cell_object_idxs) {
  const unsigned object_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (object_idx >= object_count)
    return;

  int2 first_cell, last_cell;
  object_overlapping_cells(objects[object_idx].position, first_cell, last_cell);

  // Loop over the cells an object overlaps and put the object's index in the
  // first available slot of each cell.
  for (int grid_row = first_cell.y; grid_row <= last_cell.y; grid_row++) {
    for (int grid_col = first_cell.x; grid_col <= last_cell.x; grid_col++) {
      const int cell = grid_row * GRID_WIDTH + grid_col;
      // Use atomicAdd to prevent data races.
      const int new_slot = atomicAdd(&cell_next_slot[cell], 1);
      // Do not overflow the cell if it's full.
      if (new_slot < cell_slice_starts[cell + 1])
        cell_object_idxs[new_slot] = object_idx;
    }
  }
}

// --- Simulation kernels ---

// Applies bounce and friction for an object colliding with the simulation
// bounding walls.
__device__ void apply_wall_collision(float2 &velocity,
                                     const float2 separation) {
  // If the object is already moving away from the wall, no need to apply
  // bounce.
  const float separation_factor = dot2(velocity, separation);
  if (separation_factor >= 0.0f)
    return;

  // Do not apply bounce to slow moving objects, or they will never settle.
  const float bounciness =
      separation_factor < -BOUNCE_SPEED_THRESHOLD ? BOUNCINESS : 0.0f;
  const float separation_change = -(1.0f + bounciness) * separation_factor;

  velocity.x += separation.x * separation_change;
  velocity.y += separation.y * separation_change;

  // Apply friction along the tangent vector.
  const float2 tangent = make_float2(-separation.y, separation.x);
  const float max_friction_change = FRICTION * separation_change;
  const float friction_change = clampf(
      -dot2(velocity, tangent), -max_friction_change, max_friction_change);

  velocity.x += tangent.x * friction_change;
  velocity.y += tangent.y * friction_change;
}

// Checks if an object is hitting the wall, and applies bounce and friction if
// so.
__device__ void handle_wall_collision(float2 &position, float2 &velocity) {
  if (position.x < OBJECT_RADIUS) {
    position.x = OBJECT_RADIUS;
    apply_wall_collision(velocity, make_float2(1.0f, 0.0f));
  } else if (position.x > SIM_WIDTH - OBJECT_RADIUS) {
    position.x = SIM_WIDTH - OBJECT_RADIUS;
    apply_wall_collision(velocity, make_float2(-1.0f, 0.0f));
  }

  if (position.y < OBJECT_RADIUS) {
    position.y = OBJECT_RADIUS;
    apply_wall_collision(velocity, make_float2(0.0f, 1.0f));
  } else if (position.y > SIM_HEIGHT - OBJECT_RADIUS) {
    position.y = SIM_HEIGHT - OBJECT_RADIUS;
    apply_wall_collision(velocity, make_float2(0.0f, -1.0f));
  }
}

// Called in each simulation step and uses one thread per object. Uses the
// object grid data to efficiently loop through nearby objects and perform
// physics calculations.
__global__ void physics_step(const SimObject *const objects_in,
                             SimObject *const objects_out,
                             const unsigned object_count, const float dt,
                             const int *const cell_slice_starts,
                             const int *const cell_object_idxs) {
  const unsigned object_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (object_index >= object_count)
    return;

  SimObject object = objects_in[object_index];

  const float2 start_position = object.position;

  float2 velocity = object.velocity;

  // Velocity changes are accumulated over the loop. The collision velocity is
  // caused by collisions, and the separation velocity is caused by objects
  // overlapping.
  float2 collision_velocity_change = make_float2(0.0f, 0.0f);
  float2 separation_velocity_change = make_float2(0.0f, 0.0f);

  int collision_count = 0;
  float max_separation_speed = 0.0f;

  // Calculate the range of cells in the object grid the circle is overlapping.
  // This allows efficient iteration of nearby objects.
  int2 first_cell, last_cell;
  object_overlapping_cells(start_position, first_cell, last_cell);

  for (int grid_row = first_cell.y; grid_row <= last_cell.y; grid_row++) {
    for (int grid_col = first_cell.x; grid_col <= last_cell.x; grid_col++) {
      const int cell = grid_row * GRID_WIDTH + grid_col;

      for (int slot = cell_slice_starts[cell];
           slot < cell_slice_starts[cell + 1]; slot++) {
        const int other_idx = cell_object_idxs[slot];
        if (other_idx == (int)object_index)
          continue;

        const SimObject other = objects_in[other_idx];

        // Instead of looping over all overlapping cells shared by the two
        // objects, choose the first one they both share.
        if (first_shared_cell(start_position, other.position) != cell)
          continue;

        // Compute the separation vector and overlap depth if the objects
        // overlap.
        float2 separation;
        float depth;
        if (!circles_overlap(object.position, other.position, separation,
                             depth))
          continue;

        // Adjust the position to separate the objects. Separations from
        // multiple neighbors will combine with multiple loop iterations.
        object.position.x -= 0.5f * separation.x * depth;
        object.position.y -= 0.5f * separation.y * depth;

        // Velocity in the direction of separation.
        const float2 relative_velocity = make_float2(
            other.velocity.x - velocity.x, other.velocity.y - velocity.y);
        const float separation_speed = dot2(relative_velocity, separation);

        // We want the separation to happen according to a factor of
        // SEPARATION_SPEED.
        const float wanted_separation_speed = fminf(
            SEPARATION_SPEED * fmaxf(depth - SEPARATION_TOLERANCE, 0.0f) / dt,
            MAX_SEPARATION_SPEED);

        // If we are already separating, no need to apply friction or bounce.
        if (separation_speed - wanted_separation_speed >= 0.0f)
          continue;

        // Accumulate collision counter and maximum separation speed.
        collision_count++;
        max_separation_speed =
            fmaxf(max_separation_speed, wanted_separation_speed);

        // Do not apply bounce to slow moving objects, or they will never
        // settle.
        const float bounciness =
            separation_speed < -BOUNCE_SPEED_THRESHOLD ? BOUNCINESS : 0.0f;

        // We assume both objects have equal mass, and therefore share half of
        // the velocity change.
        const float collision_change =
            -(1.0f + bounciness) * separation_speed * 0.5f;
        const float separation_change =
            (1.0f + bounciness) * wanted_separation_speed * 0.5f;

        collision_velocity_change.x -= separation.x * collision_change;
        collision_velocity_change.y -= separation.y * collision_change;
        separation_velocity_change.x -= separation.x * separation_change;
        separation_velocity_change.y -= separation.y * separation_change;

        // Apply friction along the tangent vector.
        const float2 tangent = make_float2(-separation.y, separation.x);
        const float max_friction_change = FRICTION * collision_change;
        const float friction_change =
            clampf(-0.5f * dot2(relative_velocity, tangent),
                   -max_friction_change, max_friction_change);

        collision_velocity_change.x -= tangent.x * friction_change;
        collision_velocity_change.y -= tangent.y * friction_change;
      }
    }
  }

  // If we collided with more than one other object, we need to scale down the
  // collision velocity change by a factor of the number of objects. Otherwise a
  // large number of collisions will result in an explosion. This also makes
  // sense if you think about conservation of momentum.
  if (collision_count > 1) {
    const float share = COLLISION_SHARE / (float)collision_count;
    collision_velocity_change = scale2(collision_velocity_change, share);
  }

  // On the other hand, for the separation speed, we don't divide by the number
  // of objects. If we did, the pile would slowly compress down into a sheet at
  // the bottom.
  const float separation_speed = magnitude2(separation_velocity_change);

  // Scale the separation velocity down to match the max separation speed if it
  // exceeds it.
  if (separation_speed > max_separation_speed && separation_speed > 1.0e-6f)
    separation_velocity_change = scale2(
        separation_velocity_change, max_separation_speed / separation_speed);

  // Apply accumulated velocity changes.
  velocity.x += collision_velocity_change.x + separation_velocity_change.x;
  velocity.y += collision_velocity_change.y + separation_velocity_change.y;

  // Apply the new position, but clamp the overall separation to prevent
  // explosions.
  object.position.x = start_position.x +
                      clampf(object.position.x - start_position.x,
                             -MAX_SEPARATION_PER_STEP, MAX_SEPARATION_PER_STEP);
  object.position.y = start_position.y +
                      clampf(object.position.y - start_position.y,
                             -MAX_SEPARATION_PER_STEP, MAX_SEPARATION_PER_STEP);

  // Apply gravity and damping.
  velocity.y += GRAVITY * dt;

  // Damping factor is per step instead of per time, so we convert dt into
  // steps.
  const float damping = powf(DAMPING_PER_STEP, dt / TIME_STEP);
  velocity.x *= damping;
  velocity.y *= damping;

  // Apply speed limit (terminal velocity).
  const float speed_squared = dot2(velocity, velocity);
  if (speed_squared > MAX_SPEED * MAX_SPEED)
    velocity = scale2(velocity, MAX_SPEED / sqrtf(speed_squared));

  // Calculate new position by integrating velocity.
  float2 new_position = make_float2(object.position.x + velocity.x * dt,
                                    object.position.y + velocity.y * dt);

  // Calculate wall bounce and friction based on the object's velocity.
  float2 wall_velocity = velocity;
  handle_wall_collision(new_position, wall_velocity);

  // Calculate the new velocity based on how far the object actually moved,
  // after all collisions have been processed.
  float2 actual_velocity = make_float2(
      (new_position.x - start_position.x) / dt + (wall_velocity.x - velocity.x),
      (new_position.y - start_position.y) / dt +
          (wall_velocity.y - velocity.y));

  // The intended speed is the computed speed based on collisions and gravity.
  // The actual speed is what is implied by the object's actual movement. If the
  // actual speed is greater than the intended speed, it means the object's
  // movement was dominated by collisions, and therefore the object is in the
  // pile and should settle down to avoid squashing and explosions. So we scale
  // the velocity down.
  const float intended_speed = magnitude2(velocity);
  const float actual_speed = magnitude2(actual_velocity);
  if (actual_speed > intended_speed && actual_speed > 1.0e-6f)
    actual_velocity = scale2(actual_velocity, intended_speed / actual_speed);

  // Apply the new values to the output buffer.
  object.position = new_position;
  object.velocity = actual_velocity;
  objects_out[object_index] = object;
}

// Draws objects to the Raylib texture through CUDA OpenGL interop. One thread
// per pixel. Each pixel uses the object grid data to efficiently look up
// candidate objects, then loops through candidates until a hit is found, at
// which point the object's color is written to that pixel.
__global__ void draw_objects(const cudaSurfaceObject_t surface,
                             const SimObject *const objects,
                             const unsigned width, const unsigned height,
                             const SimCamera &camera,
                             const int *const cell_slice_starts,
                             const int *const cell_object_idxs) {
  const unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  // Compute center of pixel in world space.
  const float px = camera.x + (x + 0.5f) / camera.zoom;
  const float py = camera.y + (y + 0.5f) / camera.zoom;

  // Draw the wall the same thickness regardless of camera zoom.
  const float wall_thickness = 3.0f / camera.zoom;
  const uchar4 outside_color = make_uchar4(12, 14, 18, 255);
  const uchar4 wall_color = make_uchar4(64, 70, 82, 255);
  const uchar4 inside_color = make_uchar4(20, 24, 32, 255);

  const bool inside_box =
      px >= 0.0f && px <= SIM_WIDTH && py >= 0.0f && py <= SIM_HEIGHT;
  const bool is_wall =
      px >= -wall_thickness && px <= SIM_WIDTH + wall_thickness &&
      py >= -wall_thickness && py <= SIM_HEIGHT + wall_thickness;

  // Base pixel color if no object is hit.
  uchar4 pixel_color =
      inside_box ? inside_color : (is_wall ? wall_color : outside_color);

  if (inside_box) {
    const int cell = cell_at_position(make_float2(px, py));
    // Loop over objects in the grid cell and break on the first hit.
    for (int slot = cell_slice_starts[cell]; slot < cell_slice_starts[cell + 1];
         slot++) {
      const SimObject object = objects[cell_object_idxs[slot]];
      const float dx = px - object.position.x;
      const float dy = py - object.position.y;

      if (dx * dx + dy * dy <= OBJECT_RADIUS * OBJECT_RADIUS) {
        pixel_color = object.color;
        break;
      }
    }
  }

  // Write the pixel to the texture.
  surf2Dwrite(pixel_color, surface, x * sizeof(uchar4), y);
}

// --- Host utility functions ---

// Utility function to exit upon CUDA errors.
void check_cuda(cudaError err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}

// Utility function to exit upon cuRAND errors.
void check_curand(curandStatus_t status) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuRAND error: %d\n", (int)status);
    std::exit(1);
  }
}

// Utility function to set up CUDA/OpenGL interop.
bool setup_cuda_gl() {
  constexpr unsigned MAX_GL_DEVICES = 8;
  unsigned gl_device_count;
  int gl_devices[MAX_GL_DEVICES];
  check_cuda(cudaGLGetDevices(&gl_device_count, gl_devices, MAX_GL_DEVICES,
                              cudaGLDeviceListAll));
  if (gl_device_count == 0)
    return false;
  check_cuda(cudaSetDevice(gl_devices[0]));
  return true;
}

// Creates an initialized sim camera centered on the simulation bounding box.
SimCamera setup_camera() {
  const float small_dim = std::min((float)GetScreenWidth() / SIM_WIDTH,
                                   (float)GetScreenHeight() / SIM_HEIGHT);
  const float zoom = std::clamp(small_dim, MIN_ZOOM, MAX_ZOOM);

  const float x = (SIM_WIDTH - (float)GetScreenWidth() / zoom) * 0.5f;
  const float y = (SIM_HEIGHT - (float)GetScreenHeight() / zoom) * 0.5f;

  return {x, y, zoom};
}

// --- Simulation state ---

// Returns a vector of initialized sim objects randomly scattered in a grid-like
// pattern.
std::vector<SimObject>
create_starting_objects(const unsigned long long random_offset) {
  std::vector<float> points(QUASIRANDOM_DIMENSIONS * OBJECT_COUNT);

  // Generate quasirandom data for object scattering.
  curandGenerator_t generator;
  check_curand(
      curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
  check_curand(curandSetQuasiRandomGeneratorDimensions(generator,
                                                       QUASIRANDOM_DIMENSIONS));
  check_curand(curandSetGeneratorOffset(generator, random_offset));
  check_curand(curandGenerateUniform(generator, points.data(),
                                     QUASIRANDOM_DIMENSIONS * OBJECT_COUNT));
  check_curand(curandDestroyGenerator(generator));

  // The random data is formatted in a non-interleaved way.
  const float *offsets_x = points.data();
  const float *offsets_y = points.data() + OBJECT_COUNT;

  // Create a grid of objects and use the quasirandom values to offset them.
  const unsigned grid_cols =
      std::sqrt(OBJECT_COUNT * SIM_WIDTH / SIM_HEIGHT) + 0.5f;
  const unsigned grid_rows = ceil_div(OBJECT_COUNT, grid_cols);
  const float cell_width = SIM_WIDTH / grid_cols;
  const float cell_height = SIM_HEIGHT / grid_rows;
  // Maintain a small gap between neighboring objects.
  const float gap_x = std::max(cell_width - DOUBLE_RADIUS - 0.5f, 0.0f);
  const float gap_y = std::max(cell_height - DOUBLE_RADIUS - 0.5f, 0.0f);

  std::vector<SimObject> objects(OBJECT_COUNT);
  for (unsigned i = 0; i < OBJECT_COUNT; i++) {
    const float x =
        ((i % grid_cols) + 0.5f) * cell_width + (offsets_x[i] - 0.5f) * gap_x;
    const float y =
        ((i / grid_cols) + 0.5f) * cell_height + (offsets_y[i] - 0.5f) * gap_y;

    // Use a deterministic random color based on index.
    const uchar4 color = make_uchar4(70 + (i * 37) % 170, 80 + (i * 61) % 160,
                                     100 + (i * 23) % 150, 255);

    objects[i] = {make_float2(x, y), make_float2(0.0f, 0.0f), color};
  }

  return objects;
}

// Creates the initial state of the simulation with randomly scattered starting
// objects allocated on the GPU with a double buffer.
SimState create_simulation() {
  SimState sim;
  sim.object_block_count = ceil_div(OBJECT_COUNT, THREADS_PER_BLOCK);
  sim.random_offset = 0;
  sim.cell_block_count = ceil_div(CELL_COUNT, THREADS_PER_BLOCK);

  std::vector<SimObject> objects = create_starting_objects(sim.random_offset);

  check_cuda(cudaMalloc(&sim.objects_a, OBJECT_COUNT * sizeof(*sim.objects_a)));
  check_cuda(cudaMalloc(&sim.objects_b, OBJECT_COUNT * sizeof(*sim.objects_b)));
  check_cuda(cudaMemcpy(sim.objects_a, objects.data(),
                        OBJECT_COUNT * sizeof(*sim.objects_a),
                        cudaMemcpyHostToDevice));

  // We no longer need the object data allocated on the host, so free it.
  std::vector<SimObject>().swap(objects);

  // Allocate space for the object grid data.
  check_cuda(cudaMalloc(&sim.cell_object_counts, CELL_COUNT * sizeof(int)));
  // The +1 is because the last element acts as a sentinel.
  check_cuda(
      cudaMalloc(&sim.cell_slice_starts, (CELL_COUNT + 1) * sizeof(int)));
  check_cuda(cudaMalloc(&sim.cell_next_slot, CELL_COUNT * sizeof(int)));
  check_cuda(
      cudaMalloc(&sim.cell_object_idxs, MAX_CELL_CAPACITY * sizeof(int)));
  check_cuda(cudaMalloc(&sim.block_sums, sim.cell_block_count * sizeof(int)));

  return sim;
}

// Populates the grid data to allow efficient lookup of objects in physics
// calculations.
void build_object_grid(SimState &sim) {
  // Clear the cell object counts.
  zero_cell_object_counts<<<sim.cell_block_count, THREADS_PER_BLOCK>>>(
      sim.cell_object_counts);

  // Compute the number of objects in each cell.
  compute_cell_object_counts<<<sim.object_block_count, THREADS_PER_BLOCK>>>(
      sim.objects_a, OBJECT_COUNT, sim.cell_object_counts);

  // Compute the cell data indices using shared memory, block by block.
  compute_blockwise_cell_slice_starts<<<sim.cell_block_count,
                                        THREADS_PER_BLOCK>>>(
      sim.cell_object_counts, sim.cell_slice_starts, sim.block_sums);

  // Combine previous results to get final cell data indices.
  compute_cell_slice_starts<<<1, THREADS_PER_BLOCK>>>(
      sim.block_sums, sim.cell_block_count, sim.cell_slice_starts);

  // Apply the previously calculated results to the cell_slice_starts array, as
  // well as initializing the cells' next slot indices.
  populate_cell_slice_starts<<<sim.cell_block_count, THREADS_PER_BLOCK>>>(
      sim.cell_slice_starts, sim.cell_next_slot, sim.block_sums);

  // Finally, place the object indices in their containing cells' slices.
  fill_cell_object_idxs<<<sim.object_block_count, THREADS_PER_BLOCK>>>(
      sim.objects_a, OBJECT_COUNT, sim.cell_slice_starts, sim.cell_next_slot,
      sim.cell_object_idxs);

  check_cuda(cudaGetLastError());
}

// Resets the simulation to its starting state (but with a new random offset for
// object placement). Called when the R key is pressed.
void reset_simulation(SimState &sim) {
  sim.random_offset += QUASIRANDOM_DIMENSIONS * OBJECT_COUNT;

  const std::vector<SimObject> objects =
      create_starting_objects(sim.random_offset);
  check_cuda(cudaMemcpy(sim.objects_a, objects.data(),
                        OBJECT_COUNT * sizeof(*sim.objects_a),
                        cudaMemcpyHostToDevice));
}

// Runs one physics step, rebuilding the object grid data and moving objects.
void step_simulation(SimState &sim) {
  build_object_grid(sim);

  physics_step<<<sim.object_block_count, THREADS_PER_BLOCK>>>(
      sim.objects_a, sim.objects_b, OBJECT_COUNT, TIME_STEP,
      sim.cell_slice_starts, sim.cell_object_idxs);
  check_cuda(cudaGetLastError());

  std::swap(sim.objects_a, sim.objects_b);
}

// Frees the simulation's GPU allocations.
void destroy_simulation(SimState &sim) {
  check_cuda(cudaFree(sim.objects_a));
  check_cuda(cudaFree(sim.objects_b));
  check_cuda(cudaFree(sim.cell_object_counts));
  check_cuda(cudaFree(sim.cell_slice_starts));
  check_cuda(cudaFree(sim.cell_next_slot));
  check_cuda(cudaFree(sim.cell_object_idxs));
  check_cuda(cudaFree(sim.block_sums));
}

// --- Frame handlers ---

// Called each frame. Handles camera panning and zooming.
void handle_camera(Vector2 &last_mouse_pos, SimCamera &camera) {
  const Vector2 mouse_pos = GetMousePosition();

  // Zoom in and out with the scroll wheel.
  const float scroll = GetMouseWheelMove();
  if (scroll != 0.0f) {
    const float anchor_x = camera.x + mouse_pos.x / camera.zoom;
    const float anchor_y = camera.y + mouse_pos.y / camera.zoom;

    const float new_zoom = camera.zoom * std::pow(ZOOM_STRENGTH, scroll);
    camera.zoom = std::clamp(new_zoom, MIN_ZOOM, MAX_ZOOM);

    camera.x = anchor_x - mouse_pos.x / camera.zoom;
    camera.y = anchor_y - mouse_pos.y / camera.zoom;
  }

  // Use the left mouse button to pan the camera.
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    camera.x -= (mouse_pos.x - last_mouse_pos.x) / camera.zoom;
    camera.y -= (mouse_pos.y - last_mouse_pos.y) / camera.zoom;
  }

  // Constrain the camera so at least half of the simulation bounding box is in
  // frame.
  const float view_width = (float)GetScreenWidth() / camera.zoom;
  const float view_height = (float)GetScreenHeight() / camera.zoom;

  const float keep_width = 0.5f * std::min(view_width, SIM_WIDTH);
  const float keep_height = 0.5f * std::min(view_height, SIM_HEIGHT);

  camera.x =
      std::clamp(camera.x, keep_width - view_width, SIM_WIDTH - keep_width);
  camera.y =
      std::clamp(camera.y, keep_height - view_height, SIM_HEIGHT - keep_height);

  // Keep the last mouse position up to date.
  last_mouse_pos = mouse_pos;
}

// Called each frame. Uses CUDA OpenGL interop to draw to a Raylib texture, then
// uses Raylib to render the texture.
void draw_simulation(SimTexture &target, const SimState &sim,
                     const SimCamera &camera) {
  check_cuda(cudaGraphicsMapResources(1, &target.resource, 0));

  cudaArray_t array;
  check_cuda(
      cudaGraphicsSubResourceGetMappedArray(&array, target.resource, 0, 0));

  cudaResourceDesc resource_desc = {};
  resource_desc.resType = cudaResourceTypeArray;
  resource_desc.res.array.array = array;

  cudaSurfaceObject_t surface;
  check_cuda(cudaCreateSurfaceObject(&surface, &resource_desc));

  // Invoke the kernel to draw objects (one thread per pixel).
  constexpr unsigned BLOCK_SIDE_LEN = 16;
  constexpr dim3 BLOCK_DIM(BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
  const dim3 grid_dim(ceil_div(target.width, BLOCK_SIDE_LEN),
                      ceil_div(target.height, BLOCK_SIDE_LEN));
  draw_objects<<<grid_dim, BLOCK_DIM>>>(
      surface, sim.objects_a, (unsigned)target.width, (unsigned)target.height,
      camera, sim.cell_slice_starts, sim.cell_object_idxs);
  check_cuda(cudaGetLastError());

  // Unmap the texture so it can be used by Raylib.
  check_cuda(cudaGraphicsUnmapResources(1, &target.resource, 0));
  check_cuda(cudaDestroySurfaceObject(surface));

  // Use Raylib to draw the prepared texture.
  DrawTexture(target.texture, 0, 0, WHITE);
}

// Called each frame. Draws the text in the top left.
void draw_hud() {
  DrawText(APPLICATION_NAME, 12, 12, 20, RAYWHITE);
  DrawText(TextFormat("%u objects - drag to pan, scroll to zoom, R to reset",
                      OBJECT_COUNT),
           12, 36, 16, LIGHTGRAY);
  DrawText(TextFormat("%i FPS", GetFPS()), 12, 58, 16, LIGHTGRAY);
}

// Creates an OpenGL texture and registers it with CUDA.
void create_texture(SimTexture &target, const int width, const int height) {
  const Image image = GenImageColor(width, height, BLACK);
  target.texture = LoadTextureFromImage(image);
  UnloadImage(image);
  target.width = width;
  target.height = height;
  check_cuda(cudaGraphicsGLRegisterImage(
      &target.resource, target.texture.id, GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

// Unregisters the texture from CUDA and frees its memory.
void destroy_texture(SimTexture &target) {
  check_cuda(cudaGraphicsUnregisterResource(target.resource));
  UnloadTexture(target.texture);
}

// Called each frame.
void frame(Vector2 &last_mouse_pos, SimCamera &camera, double &unsimulated_time,
           SimTexture &target, SimState &sim) {
  // Accumulate the time not yet simulated.
  const float frame_time = std::min(GetFrameTime(), MAX_FRAME_TIME);
  unsimulated_time += frame_time;

  // When the window is resized, we must recreate the texture.
  const int window_width = GetScreenWidth();
  const int window_height = GetScreenHeight();
  if (window_width > 0 && window_height > 0 &&
      (window_width != target.width || window_height != target.height)) {
    destroy_texture(target);
    create_texture(target, window_width, window_height);
  }

  handle_camera(last_mouse_pos, camera);

  // Reset the simulation when the R key is pressed.
  if (IsKeyPressed(KEY_R))
    reset_simulation(sim);

  // Step the simulation forward for the accumulated time.
  while (unsimulated_time >= TIME_STEP) {
    step_simulation(sim);
    unsimulated_time -= TIME_STEP;
  }

  // step_simulation rebuilds the grid before moving objects, so we need to
  // rebuild it once more after the last move because draw_simulation depends on
  // the grid for efficient lookup.
  build_object_grid(sim);

  BeginDrawing();
  ClearBackground(BLACK);

  draw_simulation(target, sim, camera);
  draw_hud();

  EndDrawing();
}

int main() {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, APPLICATION_NAME);

  // Set up CUDA/OpenGL interop.
  if (!setup_cuda_gl()) {
    std::fprintf(stderr,
                 "error: no CUDA device associated with OpenGL context\n");
    CloseWindow();
    return 1;
  }

  // Initialize the texture that CUDA draws into.
  SimTexture texture;
  create_texture(texture, INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);

  // Create initial simulation state on the GPU.
  SimState sim = create_simulation();

  // State for each frame.
  SimCamera camera = setup_camera();
  Vector2 last_mouse_pos = GetMousePosition();
  double unsimulated_time = 0.0;

  // Main window loop.
  while (!WindowShouldClose()) {
    frame(last_mouse_pos, camera, unsimulated_time, texture, sim);
  }

  destroy_texture(texture);
  destroy_simulation(sim);

  CloseWindow();
}
