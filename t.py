
import carla
c = carla.Client("localhost", 2000)
c.set_timeout(2.0)
w = c.get_world()
vs = w.get_actors().filter("vehicle.*")
print("num_vehicles:", len(vs))
for v in vs:
    print(v.id, v.type_id, v.attributes.get("role_name",""))

