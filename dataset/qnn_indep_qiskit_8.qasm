OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[2];

rx(0.12) q[0];
rx(0.34) q[1];
rx(0.56) q[2];
rx(0.78) q[3];
rx(0.90) q[4];
rx(0.54) q[5];

cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[0];

ry(0.236) q[0];
ry(0.452) q[1];
ry(0.679) q[2];
ry(0.124) q[3];
ry(0.578) q[4];
ry(0.891) q[5];

rz(0.654) q[0];
rz(0.248) q[1];
rz(0.743) q[2];
rz(0.419) q[3];
rz(0.902) q[4];
rz(0.567) q[5];

cx q[0],q[2];
cx q[1],q[3];
cx q[2],q[4];
cx q[3],q[5];
cx q[4],q[0];
cx q[5],q[1];

ry(0.327) q[0];
ry(0.831) q[1];
ry(0.492) q[2];
ry(0.215) q[3];
ry(0.738) q[4];
ry(0.946) q[5];

rz(0.124) q[0];
rz(0.369) q[1];
rz(0.582) q[2];
rz(0.791) q[3];
rz(0.425) q[4];
rz(0.638) q[5];

cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];

barrier q;
measure q[6] -> meas[0];
measure q[7] -> meas[1];
