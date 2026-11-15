#include <algorithm>

#include <raylib.h>

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

// Represents the state of the application's 2D camera, which allows the user to
// pan and zoom.
struct SimCamera {
  float x, y, zoom;
};

// Creates a camera centered on the bounding box of the simulation.
SimCamera setup_camera() {
  const float small_dim = std::min((float)GetScreenWidth() / SIM_WIDTH,
                                   (float)GetScreenHeight() / SIM_HEIGHT);
  const float zoom = std::clamp(small_dim, MIN_ZOOM, MAX_ZOOM);

  const float x = (SIM_WIDTH - (float)GetScreenWidth() / zoom) * 0.5f;
  const float y = (SIM_HEIGHT - (float)GetScreenHeight() / zoom) * 0.5f;

  return {x, y, zoom};
}

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

void draw_bounds(const SimCamera &camera) {
  const Rectangle bounds = {(0.0f - camera.x) * camera.zoom, //
                            (0.0f - camera.y) * camera.zoom, //
                            SIM_WIDTH * camera.zoom,         //
                            SIM_HEIGHT * camera.zoom};
  DrawRectangleRec(bounds, Color{20, 24, 32, 255});
  DrawRectangleLinesEx(bounds, 3.0f, Color{64, 70, 82, 255});
}

void draw_hud() {
  DrawText(APPLICATION_NAME, 12, 12, 20, RAYWHITE);
  DrawText("0 objects - drag to pan, scroll to zoom", 12, 36, 16, LIGHTGRAY);
  DrawText(TextFormat("%i FPS", GetFPS()), 12, 58, 16, LIGHTGRAY);
}

// Called each frame.
void frame(Vector2 &last_mouse_pos, SimCamera &camera) {
  handle_camera(last_mouse_pos, camera);

  BeginDrawing();

  ClearBackground(BLACK);

  draw_bounds(camera);
  draw_hud();

  EndDrawing();
}

int main() {
  // Initialize window and make it resizeable.
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, APPLICATION_NAME);

  SimCamera camera = setup_camera();
  Vector2 last_mouse_pos = GetMousePosition();

  // Main window loop.
  while (!WindowShouldClose()) {
    frame(last_mouse_pos, camera);
  }

  CloseWindow();
}
