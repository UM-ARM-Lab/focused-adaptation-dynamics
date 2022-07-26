from dm_control import  viewer
from dm_control.viewer import application

if __name__ == '__main__':
    app = application.Application(title='viewer', width=1024, height=768)
    app._load_environment(zoom_to_scene=True)


    def tick():
        app._viewport.set_size(1024, 768)
        app._tick()
        app._renderer.pixels.shape


    app._window.event_loop(tick_func=tick)
    app._window.close()
