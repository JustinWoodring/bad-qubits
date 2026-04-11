
h q[0];
cx q[0],q[7];

h q[1];
cx q[1],q[8];

h q[2];
cx q[2],q[9];

h q[3];
cx q[3],q[10];

h q[4];
cx q[4],q[11];

h q[5];
cx q[5],q[12];

h q[6];
cx q[6],q[13];

x q[0];


x q[3];
x q[4];

x q[6];

z q[0];
z q[1];

z q[3];

z q[5];


cx q[0],q[7];
h q[0];

cx q[1],q[8];
h q[1];

cx q[2],q[9];
h q[2];

cx q[3],q[10];
h q[3];

cx q[4],q[11];
h q[4];

cx q[5],q[12];
h q[5];

cx q[6],q[13];
h q[6];

barrier q;
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
