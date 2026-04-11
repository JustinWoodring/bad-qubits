OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg meas[3];


t q[0];

barrier q;
measure q[0] -> meas[0];

x q[1];

t q[1];

measure q[1] -> meas[1];

h q[2];

t q[2];

h q[2];
measure q[2] -> meas[2];
