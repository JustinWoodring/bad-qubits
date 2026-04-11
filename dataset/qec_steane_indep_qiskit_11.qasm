
h q[0];

h q[1];
h q[2];
h q[3];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[0],q[5];
cx q[1],q[5];
cx q[3],q[5];
cx q[0],q[6];
cx q[2],q[6];
cx q[3],q[6];

x q[4];

h q[7];
h q[8];
h q[9];

cx q[0],q[7];
cx q[2],q[7];
cx q[4],q[7];
cx q[6],q[7];

cx q[1],q[8];
cx q[2],q[8];
cx q[5],q[8];
cx q[6],q[8];

cx q[3],q[9];
cx q[4],q[9];
cx q[5],q[9];
cx q[6],q[9];

h q[7];
h q[8];
h q[9];
measure q[7] -> syndrome[0];
measure q[8] -> syndrome[1];
measure q[9] -> syndrome[2];

h q[10];

cx q[10],q[0];
cx q[10],q[1];
cx q[10],q[3];

h q[10];
measure q[10] -> syndrome[3];

x q[4];

measure q[0] -> data[0];
measure q[1] -> data[1];
measure q[2] -> data[2];
measure q[3] -> data[3];
measure q[4] -> data[4];
measure q[5] -> data[5];
measure q[6] -> data[6];
