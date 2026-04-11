
h q[0];
h q[1];
h q[2];
h q[3];

x q[0];
x q[2];
ccx q[0],q[2],q[4];
x q[0];
x q[2];

x q[2];
ccx q[0],q[2],q[5];
x q[2];

x q[0];
ccx q[0],q[2],q[6];
x q[0];

ccx q[0],q[2],q[7];

h q[0];
h q[1];
h q[2];
h q[3];

barrier q;
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
