from __future__ import print_function
import tensorflow as tf

#node1 = tf.constant(3.0, dtype=tf.float32)
#node2 = tf.constant(4.0)
#node3 = tf.add(node1, node2)

#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#adder_node = a+b
#add_and_triple = adder_node * 3

W = tf.Variable([0.3], dtype=tf.float32)
B = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + B

square_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(square_deltas)

fixW = tf.assign(W, [-1])
fixB = tf.assign(B, [1])

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
sess.run([fixW, fixB])
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
#print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

#print(sess.run(add_and_triple, {a: 3, b: 4.5}))

#print(sess.run(adder_node, {a: 3, b: 4.5}))
#print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
# print("node3:",node3)
# print("sess.run(node3)", sess.run(node3))


