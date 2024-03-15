import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Push to the left of a list
r.lpush('mylist', 'one')
r.lpush('mylist', 'two')
r.lpush('mylist', 'three')

# Retrieve elements from the list
elements = r.lrange('mylist', 0, -1)
print(elements)  # Output: [b'three', b'two', b'one']