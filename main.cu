#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_gl_interop.h>
#include <curand.h>

#include <raylib.h>

// Macro for nonnegative integer ceiling division.
#define ceil_div(a, b) (((a) + ((b) - 1)) / (b))

// --- Window & camera settings ---

constexpr unsigned INITIAL_WINDOW_WIDTH = 1280;
constexpr unsigned INITIAL_WINDOW_HEIGHT = 720;

constexpr float MIN_ZOOM = 0.2f;
constexpr float MAX_ZOOM = 12.0f;
constexpr float ZOOM_STRENGTH = 1.1f;

constexpr char APPLICATION_NAME[] = "CUDA + Raylib Rigidbody Simulation";

// --- Simulation settings ---

constexpr float SIM_WIDTH = 2400.0f;
constexpr float SIM_HEIGHT = 1350.0f;

constexpr unsigned OBJECT_COUNT = 3000;
constexpr float OBJECT_RADIUS = 8.0f;

// Run the physics steps at a fixed rate. If a frame takes longer than one
// physics step, allow it to run at most 2 steps to catch up. If we allowed
// more, it could create a chain reaction of longer and longer frame times.
constexpr float PHYSICS_DT = 1.0f / 60.0f;
constexpr float MAX_FRAME_TIME = 2.0f * PHYSICS_DT;

constexpr float GRAVITY = 180.0f;
constexpr float WALL_BOUNCE = 0.3f;
constexpr float OBJECT_BOUNCE = 0.0f;

// Add a very small amount of damping; this allows the objects to settle without
// wiggling forever.
constexpr float DAMPING_PER_STEP = 0.999f;
constexpr float MAX_SPEED = 900.0f;

// Limit the correction factor so tightly packed objects don't explode.
constexpr float MAX_CORRECTION = 1.0f;

// Threads per block for both kernels.
constexpr unsigned THREADS_PER_BLOCK = 256;

// X offset, Y offset, red, green, and blue color.
constexpr unsigned QUASIRANDOM_DIMENSIONS = 5;

// --- Data structures ---

// Represents the state of the application's 2D camera, which allows the user to
// pan and zoom.
struct SimCamera {
  float x, y, zoom;
};

// Represents one ball in the simulation. Since balls are rotationally
// symmetric, no rotation is needed.
struct SimBall {
  float2 position;
  float2 velocity;
  float radius;
  uchar4 color;
};

// Represents the texture that CUDA draws the simulation into.
struct RenderTarget {
  Texture2D texture;
  cudaGraphicsResource *resource;
  unsigned width, height;
};

// Represents the state of the simulation.
struct SimState {
  SimBall *buf_a;
  SimBall *buf_b;
  unsigned block_count;
  unsigned long long rng_offset;
};

// --- Device utility functions ---

// Clamps a value between a min and max value. std::clamp can't be called in
// device code, so we implement it ourselves.
__device__ float clamp(const float value, const float min, const float max) {
  return fminf(fmaxf(value, min), max);
}

// --- Kernels ---

// Called for each other object an object might collide with. Pushes obj_a out
// of obj_b and computes its new velocity.
__device__ void handle_object_collision(const float radius, float &x, float &y,
                                        const SimBall &obj_b, float &vx,
                                        float &vy) {
  const float dx = obj_b.position.x - x;
  const float dy = obj_b.position.y - y;

  const float r = radius + obj_b.radius;
  const float sqr_dist = dx * dx + dy * dy;

  // Skip if the objects are too far apart or if they are just touching.
  if (sqr_dist >= r * r || sqr_dist < 1.0e-6f)
    return;

  const float dist = sqrtf(sqr_dist);
  const float nx = dx / dist;
  const float ny = dy / dist;
  const float overlap = r - dist;

  // All objects have the same mass, so each object gets half of the correction.
  x -= nx * overlap * 0.5f;
  y -= ny * overlap * 0.5f;

  // Calculate the speed of obj_b along the normal vector of the collision.
  const float vn = (obj_b.velocity.x - vx) * nx + (obj_b.velocity.y - vy) * ny;

  // Positive normal speed means the objects are already separating.
  if (vn >= 0.0f)
    return;

  // Apply collision impulse plus bounce. Again, since all objects have equal
  // mass, each object gets half of the impulse.
  const float impulse = -(1.0f + OBJECT_BOUNCE) * vn * 0.5f;
  vx -= nx * impulse;
  vy -= ny * impulse;
}

// Called once for each object. Handles collision with the boundary walls.
__device__ void handle_wall_collision(const SimBall &obj, float &x, float &y,
                                      const float in_vx, const float in_vy,
                                      float &vx, float &vy) {
  const float r = obj.radius;
  if (x < r) {
    x = r;
    if (in_vx < 0.0f)
      vx = -in_vx * WALL_BOUNCE;
  } else if (x > SIM_WIDTH - r) {
    x = SIM_WIDTH - r;
    if (in_vx > 0.0f)
      vx = -in_vx * WALL_BOUNCE;
  }
  if (y < r) {
    y = r;
    if (in_vy < 0.0f)
      vy = -in_vy * WALL_BOUNCE;
  } else if (y > SIM_HEIGHT - r) {
    y = SIM_HEIGHT - r;
    if (in_vy > 0.0f)
      vy = -in_vy * WALL_BOUNCE;
  }
}

// Called each simulation step. Uses one thread per object.
__global__ void step_sim(const SimBall *const objects_in,
                         SimBall *const objects_out,
                         const unsigned object_count, const float dt) {
  const unsigned thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_index >= object_count)
    return;

  SimBall obj_a = objects_in[thread_index];
  const float start_x = obj_a.position.x;
  const float start_y = obj_a.position.y;

  // Accumulate position corrections as we loop over the other objects.
  float new_x = start_x;
  float new_y = start_y;

  float new_vx = obj_a.velocity.x;
  float new_vy = obj_a.velocity.y;

  for (unsigned i = 0; i < object_count; i++) {
    if (i == thread_index)
      continue;

    handle_object_collision(obj_a.radius, new_x, new_y, objects_in[i], new_vx,
                            new_vy);
  }

  // Prevent the correction from being too high, which would cause an explosion.
  // If objects are deeply overlapping, the overlap will be corrected in
  // multiple steps.
  new_x = start_x + clamp(new_x - start_x, -MAX_CORRECTION, MAX_CORRECTION);
  new_y = start_y + clamp(new_y - start_y, -MAX_CORRECTION, MAX_CORRECTION);

  // Apply gravity and damping.
  new_vy += GRAVITY * dt;

  new_vx *= DAMPING_PER_STEP;
  new_vy *= DAMPING_PER_STEP;

  // Apply max speed (terminal velocity).
  const float sqr_speed = new_vx * new_vx + new_vy * new_vy;
  if (sqr_speed > MAX_SPEED * MAX_SPEED) {
    const float scale = MAX_SPEED / sqrtf(sqr_speed);
    new_vx *= scale;
    new_vy *= scale;
  }

  // Integrate position from new velocity.
  new_x += new_vx * dt;
  new_y += new_vy * dt;

  // Calculate the new velocity based on how far the object actually moved.
  float end_vx = (new_x - start_x) / dt;
  float end_vy = (new_y - start_y) / dt;

  // The arrival speed is the speed of the object after collisions and gravity
  // have been calculated. However, if an object is sitting in the pile and we
  // just used this velocity, gravity would continue driving it downwards,
  // creating a squishing effect. To fix this, we use the velocity implied by
  // the actual displacement (pushed speed). If that is larger, then the object
  // is dominated by collisions rather than gravity, so we prevent collisions
  // from adding speed to objects so that they settle down.
  const float arrival_speed = sqrtf(new_vx * new_vx + new_vy * new_vy);
  const float pushed_speed = sqrtf(end_vx * end_vx + end_vy * end_vy);

  if (pushed_speed > arrival_speed && pushed_speed > 1.0e-6f) {
    const float scale = arrival_speed / pushed_speed;
    end_vx *= scale;
    end_vy *= scale;
  }

  handle_wall_collision(obj_a, new_x, new_y, new_vx, new_vy, end_vx, end_vy);

  obj_a.position = make_float2(new_x, new_y);
  obj_a.velocity = make_float2(end_vx, end_vy);
  objects_out[thread_index] = obj_a;
}

// Called each simulation step, one thread per output pixel. Draws objects by
// writing pixels to a target texture. This is simple, efficient, and scales to
// a large number of objects. We could draw instanced meshes with Raylib
// instead, but it would not scale all the way to millions of objects due to
// limitations of the API and the fact that it requires copying data through the
// host. This approach scales well because it scales with the number of pixels
// and does not blow up in complexity the way meshes would if millions needed to
// be drawn.
__global__ void draw_objects(const cudaSurfaceObject_t surface,
                             const SimBall *const objects, const unsigned width,
                             const unsigned height, const SimCamera &camera) {
  const unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  // Convert center of pixel to point in world space.
  const float px = camera.x + (x + 0.5f) / camera.zoom;
  const float py = camera.y + (y + 0.5f) / camera.zoom;

  // Draw the wall in world units so it's always the same thickness regardless
  // of zoom.
  const float wall_thickness = 3.0f / camera.zoom;
  const uchar4 outside_color = make_uchar4(12, 14, 18, 255);
  const uchar4 wall_color = make_uchar4(64, 70, 82, 255);
  const uchar4 inside_color = make_uchar4(20, 24, 32, 255);

  const bool inWorld =
      px >= 0.0f && px <= SIM_WIDTH && py >= 0.0f && py <= SIM_HEIGHT;
  const bool inFrame =
      px >= -wall_thickness && px <= SIM_WIDTH + wall_thickness &&
      py >= -wall_thickness && py <= SIM_HEIGHT + wall_thickness;

  // Base pixel color based on whether it's inside, outside, or wall.
  uchar4 pixel_color =
      inWorld ? inside_color : (inFrame ? wall_color : outside_color);

  if (inWorld) {
    for (unsigned i = 0; i < OBJECT_COUNT; i++) {
      const SimBall obj = objects[i];
      const float dx = px - obj.position.x;
      const float dy = py - obj.position.y;
      // If the pixel is inside the circle, make it the color of the object.
      if (dx * dx + dy * dy <= obj.radius * obj.radius) {
        pixel_color = obj.color;
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

// Creates a camera centered on the bounding box of the simulation.
SimCamera setup_camera() {
  const float small_dim = std::min((float)GetScreenWidth() / SIM_WIDTH,
                                   (float)GetScreenHeight() / SIM_HEIGHT);
  const float zoom = std::clamp(small_dim, MIN_ZOOM, MAX_ZOOM);

  const float x = (SIM_WIDTH - (float)GetScreenWidth() / zoom) * 0.5f;
  const float y = (SIM_HEIGHT - (float)GetScreenHeight() / zoom) * 0.5f;

  return {x, y, zoom};
}

// --- Simulation state ---

// Creates a quasirandom distribution of objects inside the simulation
// boundaries.
std::vector<SimBall>
fill_objects_quasirandom(const unsigned long long rng_offset) {
  std::vector<float> points(QUASIRANDOM_DIMENSIONS * OBJECT_COUNT);

  // Generate quasirandom data of the correct dimensions.
  curandGenerator_t generator;
  check_curand(
      curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
  check_curand(curandSetQuasiRandomGeneratorDimensions(generator,
                                                       QUASIRANDOM_DIMENSIONS));
  check_curand(curandSetGeneratorOffset(generator, rng_offset));
  check_curand(curandGenerateUniform(generator, points.data(),
                                     QUASIRANDOM_DIMENSIONS * OBJECT_COUNT));
  check_curand(curandDestroyGenerator(generator));

  // The data is laid out contiguously with each dimension together.
  const float *offset_x = points.data();
  const float *offset_y = points.data() + OBJECT_COUNT;
  const float *color_r = points.data() + 2 * OBJECT_COUNT;
  const float *color_g = points.data() + 3 * OBJECT_COUNT;
  const float *color_b = points.data() + 4 * OBJECT_COUNT;

  // Use the quasirandom data to nudge the points off of their grid lines. If we
  // just used pure quasirandom points then some objects would overlap.
  const unsigned grid_cols =
      std::sqrt(OBJECT_COUNT * SIM_WIDTH / SIM_HEIGHT) + 0.5f;
  const unsigned grid_rows = ceil_div(OBJECT_COUNT, grid_cols);
  const float cell_width = SIM_WIDTH / grid_cols;
  const float cell_height = SIM_HEIGHT / grid_rows;

  // Maintain a gap between objects.
  const float gap_x = std::max(cell_width - 2.0f * OBJECT_RADIUS - 0.5f, 0.0f);
  const float gap_y = std::max(cell_height - 2.0f * OBJECT_RADIUS - 0.5f, 0.0f);

  std::vector<SimBall> objects(OBJECT_COUNT);
  for (unsigned i = 0; i < OBJECT_COUNT; i++) {
    const float x =
        ((i % grid_cols) + 0.5f) * cell_width + (offset_x[i] - 0.5f) * gap_x;
    const float y =
        ((i / grid_cols) + 0.5f) * cell_height + (offset_y[i] - 0.5f) * gap_y;

    // Use the random data to generate a unique color.
    const uchar4 color =
        make_uchar4((unsigned char)(60 + 195 * color_r[i]),
                    (unsigned char)(60 + 195 * color_g[i]),
                    (unsigned char)(80 + 175 * color_b[i]), 255);

    objects[i] = {make_float2(x, y), make_float2(0.0f, 0.0f), OBJECT_RADIUS,
                  color};
  }

  return objects;
}

// Initializes the simulation state and copies data to the GPU using a double
// buffer.
SimState create_simulation() {
  SimState sim;
  sim.block_count = ceil_div(OBJECT_COUNT, THREADS_PER_BLOCK);
  sim.rng_offset = 0;

  std::vector<SimBall> objects = fill_objects_quasirandom(sim.rng_offset);

  check_cuda(cudaMalloc(&sim.buf_a, OBJECT_COUNT * sizeof(*sim.buf_a)));
  check_cuda(cudaMalloc(&sim.buf_b, OBJECT_COUNT * sizeof(*sim.buf_b)));
  check_cuda(cudaMemcpy(sim.buf_a, objects.data(),
                        OBJECT_COUNT * sizeof(*sim.buf_a),
                        cudaMemcpyHostToDevice));

  return sim;
}

// Resets the simulation state, but creates a new RNG offset to randomize object
// positions.
void reset_simulation(SimState &sim) {
  sim.rng_offset += QUASIRANDOM_DIMENSIONS * OBJECT_COUNT;

  const std::vector<SimBall> objects = fill_objects_quasirandom(sim.rng_offset);
  check_cuda(cudaMemcpy(sim.buf_a, objects.data(),
                        OBJECT_COUNT * sizeof(*sim.buf_a),
                        cudaMemcpyHostToDevice));
}

// Runs one physics step.
void step_simulation(SimState &sim) {
  step_sim<<<sim.block_count, THREADS_PER_BLOCK>>>(sim.buf_a, sim.buf_b,
                                                   OBJECT_COUNT, PHYSICS_DT);
  check_cuda(cudaGetLastError());
  std::swap(sim.buf_a, sim.buf_b);
}

// Frees the simulation GPU buffers.
void destroy_simulation(SimState &sim) {
  check_cuda(cudaFree(sim.buf_a));
  check_cuda(cudaFree(sim.buf_b));
}

// --- Frame handlers ---

// Called each frame. Handles camera panning and zooming.
void handle_camera(Vector2 &last_mouse_pos, SimCamera &camera) {
  const Vector2 mouse_pos = GetMousePosition();

  // Zoom the camera using the scroll wheel.
  const float scroll = GetMouseWheelMove();
  if (scroll != 0.0f) {
    const float anchor_x = camera.x + mouse_pos.x / camera.zoom;
    const float anchor_y = camera.y + mouse_pos.y / camera.zoom;

    const float new_zoom = camera.zoom * std::pow(ZOOM_STRENGTH, scroll);
    camera.zoom = std::clamp(new_zoom, MIN_ZOOM, MAX_ZOOM);

    camera.x = anchor_x - mouse_pos.x / camera.zoom;
    camera.y = anchor_y - mouse_pos.y / camera.zoom;
  }

  // If the left mouse button is held, pan the camera.
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    camera.x -= (mouse_pos.x - last_mouse_pos.x) / camera.zoom;
    camera.y -= (mouse_pos.y - last_mouse_pos.y) / camera.zoom;
  }

  // Prevent the camera from panning away from the simulation bounds.
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
void draw_simulation(RenderTarget &target, const SimBall *const objects,
                     const SimCamera &camera) {
  check_cuda(cudaGraphicsMapResources(1, &target.resource, 0));

  cudaArray_t array;
  check_cuda(
      cudaGraphicsSubResourceGetMappedArray(&array, target.resource, 0, 0));

  // A surface object wraps the mapped array so the kernel can write to it.
  cudaResourceDesc resource_desc = {};
  resource_desc.resType = cudaResourceTypeArray;
  resource_desc.res.array.array = array;

  cudaSurfaceObject_t surface;
  check_cuda(cudaCreateSurfaceObject(&surface, &resource_desc));

  // Invoke the draw kernel (one thread per pixel).
  const dim3 block(16, 16);
  const dim3 grid(ceil_div(target.width, block.x),
                  ceil_div(target.height, block.y));
  draw_objects<<<grid, block>>>(surface, objects, target.width, target.height,
                                camera);
  check_cuda(cudaGetLastError());

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
void create_texture(RenderTarget &target, const unsigned width,
                    const unsigned height) {
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
void destroy_texture(RenderTarget &target) {
  check_cuda(cudaGraphicsUnregisterResource(target.resource));
  UnloadTexture(target.texture);
}

// Called each frame.
void frame(Vector2 &last_mouse_pos, SimCamera &camera, double &unsimulated_time,
           RenderTarget &target, SimState &sim) {
  // Handle accumulated time.
  const float frame_time = std::min(GetFrameTime(), MAX_FRAME_TIME);
  unsimulated_time += frame_time;

  // Resize texture if window was resized.
  const unsigned window_width = GetScreenWidth();
  const unsigned window_height = GetScreenHeight();
  if (window_width > 0 && window_height > 0 &&
      (window_width != target.width || window_height != target.height)) {
    destroy_texture(target);
    create_texture(target, window_width, window_height);
  }

  handle_camera(last_mouse_pos, camera);

  // Restart the simulation when the R key is pressed.
  if (IsKeyPressed(KEY_R))
    reset_simulation(sim);

  // Run physics steps for the accumulated time.
  while (unsimulated_time >= PHYSICS_DT) {
    step_simulation(sim);
    unsimulated_time -= PHYSICS_DT;
  }

  BeginDrawing();

  ClearBackground(BLACK);

  draw_simulation(target, sim.buf_a, camera);
  draw_hud();

  EndDrawing();
}

int main() {
  // Initialize window and make it resizeable.
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
  RenderTarget target;
  create_texture(target, INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);

  // Create simulation state on the GPU.
  SimState sim = create_simulation();

  // State for each frame.
  SimCamera camera = setup_camera();
  Vector2 last_mouse_pos = GetMousePosition();
  double unsimulated_time = 0.0;

  // Main window loop.
  while (!WindowShouldClose()) {
    frame(last_mouse_pos, camera, unsimulated_time, target, sim);
  }

  destroy_texture(target);
  destroy_simulation(sim);

  CloseWindow();

  return 0;
}
