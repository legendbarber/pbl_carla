import carla
c = carla.Client("127.0.0.1", 2000)
c.set_timeout(5.0)
print("carla_module:", carla.__file__)
print("server_version:", c.get_server_version())
try:
    print("client_version:", c.get_client_version())
except Exception as e:
    print("client_version: (err)", e)
w = c.get_world()
actors = w.get_actors()
print("actors_total:", len(actors))
print("vehicles:", len(actors.filter("vehicle.*")))