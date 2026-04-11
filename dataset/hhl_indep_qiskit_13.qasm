OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg meas[6];

x q[0];

h q[4];
h q[5];
h q[6];
h q[7];

cu1(pi/8) q[7],q[0];
cu1(pi/8) q[7],q[1];
cu1(pi/8) q[7],q[2];
cu1(pi/8) q[7],q[3];

cu1(pi/4) q[6],q[0];
cu1(pi/4) q[6],q[1];
cu1(pi/4) q[6],q[2];
cu1(pi/4) q[6],q[3];

cu1(pi/2) q[5],q[0];
cu1(pi/2) q[5],q[1];
cu1(pi/2) q[5],q[2];
cu1(pi/2) q[5],q[3];

cu1(pi) q[4],q[0];
cu1(pi) q[4],q[1];
cu1(pi) q[4],q[2];
cu1(pi) q[4],q[3];

swap q[4],q[7];
swap q[5],q[6];

h q[4];
cu1(-pi/2) q[4],q[5];
cu1(-pi/4) q[4],q[6];
cu1(-pi/8) q[4],q[7];
h q[5];
cu1(-pi/2) q[5],q[6];
cu1(-pi/4) q[5],q[7];
h q[6];
cu1(-pi/2) q[6],q[7];
h q[7];

c3x q[4],q[5],q[6],q[8];
c3x q[4],q[5],q[7],q[9];
c3x q[4],q[6],q[7],q[10];

cu3(pi/4,0,0) q[8],q[11];
cu3(pi/2,0,0) q[9],q[11];
cu3(pi,0,0) q[10],q[11];

h q[7];
cu1(pi/2) q[6],q[7];
h q[6];
cu1(pi/2) q[5],q[6];
cu1(pi/4) q[5],q[7];
h q[5];
cu1(pi/2) q[4],q[5];
cu1(pi/4) q[4],q[6];
cu1(pi/8) q[4],q[7];
h q[4];

swap q[4],q[7];
swap q[5],q[6];

cu1(-pi) q[4],q[0];
cu1(-pi) q[4],q[1];
cu1(-pi) q[4],q[2];
cu1(-pi) q[4],q[3];

cu1(-pi/2) q[5],q[0];
cu1(-pi/2) q[5],q[1];
cu1(-pi/2) q[5],q[2];
cu1(-pi/2) q[5],q[3];

cu1(-pi/4) q[6],q[0];
cu1(-pi/4) q[6],q[1];
cu1(-pi/4) q[6],q[2];
cu1(-pi/4) q[6],q[3];

cu1(-pi/8) q[7],q[0];
cu1(-pi/8) q[7],q[1];
cu1(-pi/8) q[7],q[2];
cu1(-pi/8) q[7],q[3];

h q[4];
h q[5];
h q[6];
h q[7];

barrier q;
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[11] -> meas[4];
measure q[12] -> meas[5];
