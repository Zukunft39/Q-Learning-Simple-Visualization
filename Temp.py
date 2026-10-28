Q = {}
state = (3,5)
action = {"w","a", "s", "d"}

Q[(state, "w")] = 5.0


v1 = Q.get((state, "w"))
v2 = Q.get((state, "a"), 0.0)

print(v1)  # 输出: 5.0
print(v2)  # 输出: 0.0