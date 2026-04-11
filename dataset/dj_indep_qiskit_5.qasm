OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg meas[4];

x q[4];

h q[0];
h q[1];
h q[2];
h q[3];
h q[4];

cx q[0],q[4];
cx q[2],q[4];

h q[0];
h q[1];
h q[2];
h q[3];

barrier q;
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
