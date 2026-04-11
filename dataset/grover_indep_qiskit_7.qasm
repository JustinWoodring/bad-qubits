
h q[0];
h q[1];
h q[2];

x q[3];
h q[3];

x q[2];
cx q[0],q[3];
cx q[1],q[3];
x q[2];

h q[0];
h q[1];
h q[2];
x q[0];
x q[1];
x q[2];

h q[4];
ccx q[0],q[1],q[4];
ccx q[4],q[2],q[5];
ccx q[0],q[1],q[4];
ccx q[4],q[2],q[5];
h q[4];

x q[0];
x q[1];
x q[2];
h q[0];
h q[1];
h q[2];

x q[2];
cx q[0],q[3];
cx q[1],q[3];
x q[2];

h q[0];
h q[1];
h q[2];
x q[0];
x q[1];
x q[2];

h q[6];
ccx q[0],q[1],q[6];
ccx q[6],q[2],q[5];
ccx q[0],q[1],q[6];
ccx q[6],q[2],q[5];
h q[6];

x q[0];
x q[1];
x q[2];
h q[0];
h q[1];
h q[2];

barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
