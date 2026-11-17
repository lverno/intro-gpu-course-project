#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_gl_interop.h>

#include <raylib.h>

// Macro for nonnegative integer ceiling division.
#define ceil_div(a, b) (((a) + ((b) - 1)) / (b))

// --- Window & camera settings ---

constexpr unsigned INITIAL_WINDOW_WIDTH = 1280;
constexpr unsigned INITIAL_WINDOW_HEIGHT = 720;

constexpr float MIN_ZOOM = 0.25f;
constexpr float MAX_ZOOM = 8.0f;
constexpr float ZOOM_STRENGTH = 1.1f;

constexpr char APPLICATION_NAME[] = "CUDA + Raylib Rigidbody Simulation";

// --- Simulation settings ---

constexpr float SIM_WIDTH = 1600.0f;
constexpr float SIM_HEIGHT = 900.0f;

constexpr unsigned OBJECT_COUNT = 480;
constexpr float OBJECT_RADIUS = 18.0f;

// Run the physics simulation in fixed timesteps, independent of the frame rate.
// If a frame takes longer than a physics time step, then a frame may run at
// most two physics steps. If more were allowed, it could create a chain
// reaction of slower and slower frames.
constexpr float PHYSICS_DT = 1.0f / 60.0f;
constexpr float MAX_FRAME_TIME = 2.0f * PHYSICS_DT;

constexpr float GRAVITY = 180.0f;
constexpr float WALL_BOUNCE = 0.3f;
constexpr float OBJECT_BOUNCE = 0.2f;

// Add a very small amount of damping; this allows the objects to settle without
// wiggling forever.
constexpr float DAMPING_PER_STEP = 0.999f;
constexpr float MAX_SPEED = 900.0f;

// Limit the correction factor so tightly packed objects don't explode.
constexpr float MAX_CORRECTION = 2.0f;

// Threads per block for both kernels.
constexpr unsigned THREADS_PER_BLOCK = 256;

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

  // Loop through all other objects. This is naive and does not scale well, but
  // is simple and works well for a few hundred objects.
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
      SimBall obj = objects[i];
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
  DrawText(TextFormat("%u objects - drag to pan, scroll to zoom", OBJECT_COUNT),
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
           RenderTarget &target, SimBall *&current_obj, SimBall *&next_obj,
           const unsigned block_count) {
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

  // Run physics steps for the accumulated time.
  while (unsimulated_time >= PHYSICS_DT) {
    step_sim<<<block_count, THREADS_PER_BLOCK>>>(current_obj, next_obj,
                                                 OBJECT_COUNT, PHYSICS_DT);
    check_cuda(cudaGetLastError());
    std::swap(current_obj, next_obj);
    unsimulated_time -= PHYSICS_DT;
  }

  BeginDrawing();

  ClearBackground(BLACK);
  draw_simulation(target, current_obj, camera);

  draw_hud();

  EndDrawing();
}

// Creates a vector of objects with random colors in a grid pattern.
std::vector<SimBall> create_initial_objects() {
  constexpr unsigned GRID_COLS = 24;
  constexpr unsigned GRID_ROWS = OBJECT_COUNT / GRID_COLS;

  static_assert(GRID_COLS * GRID_ROWS == OBJECT_COUNT);

  std::vector<SimBall> objects(OBJECT_COUNT);

  for (unsigned i = 0; i < OBJECT_COUNT; i++) {
    const float grid_x = ((i % GRID_COLS) + 0.5f) * (SIM_WIDTH / GRID_COLS);
    const float grid_y = ((i / GRID_COLS) + 0.5f) * (SIM_HEIGHT / GRID_ROWS);
    // Use a deterministic object color based on the index.
    const uchar4 color = make_uchar4(70 + (i * 37) % 170, 80 + (i * 61) % 160,
                                     100 + (i * 23) % 150, 255);
    objects[i] = {make_float2(grid_x, grid_y), make_float2(0.0f, 0.0f),
                  OBJECT_RADIUS, color};
  }

  return objects;
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

  // Create initial objects and copy to a double buffer on the GPU.
  std::vector<SimBall> objects = create_initial_objects();
  SimBall *buf_a;
  SimBall *buf_b;
  check_cuda(cudaMalloc(&buf_a, OBJECT_COUNT * sizeof(*buf_a)));
  check_cuda(cudaMalloc(&buf_b, OBJECT_COUNT * sizeof(*buf_b)));
  check_cuda(cudaMemcpy(buf_a, objects.data(), OBJECT_COUNT * sizeof(*buf_a),
                        cudaMemcpyHostToDevice));

  // Deallocate host object buffer as it is no longer needed.
  std::vector<SimBall>().swap(objects);

  SimBall *current_obj = buf_a;
  SimBall *next_obj = buf_b;
  const unsigned block_count = ceil_div(OBJECT_COUNT, THREADS_PER_BLOCK);

  // State for each frame.
  SimCamera camera = setup_camera();
  Vector2 last_mouse_pos = GetMousePosition();
  double unsimulated_time = 0.0;

  // Main window loop.
  while (!WindowShouldClose()) {
    frame(last_mouse_pos, camera, unsimulated_time, target, current_obj,
          next_obj, block_count);
  }

  destroy_texture(target);

  check_cuda(cudaFree(buf_a));
  check_cuda(cudaFree(buf_b));

  CloseWindow();

  return 0;
}
