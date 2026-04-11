OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg meas[4];

x q[0];
x q[1];

rx(0.123) q[0];
rx(0.234) q[1];
rx(0.345) q[2];
rx(0.456) q[3];

ry(0.567) q[0];
ry(0.678) q[1];
ry(0.789) q[2];
ry(0.890) q[3];

cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0];

rx(0.901) q[0];
rx(0.812) q[1];
rx(0.723) q[2];
rx(0.634) q[3];

ry(0.545) q[0];
ry(0.456) q[1];
ry(0.367) q[2];
ry(0.278) q[3];

cx q[0],q[2];
cx q[1],q[3];
cx q[2],q[0];
cx q[3],q[1];

rx(0.189) q[0];
rx(0.298) q[1];
rx(0.387) q[2];
rx(0.476) q[3];

ry(0.565) q[0];
ry(0.654) q[1];
ry(0.743) q[2];
ry(0.832) q[3];

barrier q;
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
