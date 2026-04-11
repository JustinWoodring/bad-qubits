measure q[0] -> mout[0];
measure q[1] -> mout[1];
ry(pi/4) q[1];
x q[0];
rx(pi/4) q[0];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
