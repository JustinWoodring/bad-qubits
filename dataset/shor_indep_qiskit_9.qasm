OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg meas[4];

h q[0];
h q[1];
h q[2];
h q[3];

x q[8];


cu1(0) q[3],q[4];
cu1(0) q[3],q[5];
cu1(pi) q[3],q[6];
cu1(pi) q[3],q[7];
cu1(pi) q[3],q[8];

cu1(0) q[2],q[4];
cu1(0) q[2],q[5];
cu1(0) q[2],q[6];
cu1(pi) q[2],q[7];
cu1(0) q[2],q[8];

cu1(0) q[1],q[4];
cu1(0) q[1],q[5];
cu1(0) q[1],q[6];
cu1(0) q[1],q[7];
cu1(0) q[1],q[8];

cu1(0) q[0],q[4];
cu1(0) q[0],q[5];
cu1(0) q[0],q[6];
cu1(0) q[0],q[7];
cu1(0) q[0],q[8];

swap q[0],q[3];
swap q[1],q[2];

h q[0];
cu1(-pi/2) q[0],q[1];
cu1(-pi/4) q[0],q[2];
cu1(-pi/8) q[0],q[3];
h q[1];
cu1(-pi/2) q[1],q[2];
cu1(-pi/4) q[1],q[3];
h q[2];
cu1(-pi/2) q[2],q[3];
h q[3];

barrier q;
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
