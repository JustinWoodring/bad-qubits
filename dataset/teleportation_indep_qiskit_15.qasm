
h q[0];
t q[0];
h q[0];

h q[1];
cx q[1],q[2];

h q[3];
cx q[3],q[4];

h q[5];
cx q[5],q[6];

h q[7];
cx q[7],q[8];

h q[9];
cx q[9],q[10];

h q[11];
cx q[11],q[12];

h q[13];
cx q[13],q[14];

cx q[0],q[1];
h q[0];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
if(meas[1]==1) x q[2];
if(meas[0]==1) z q[2];

cx q[2],q[3];
h q[2];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
if(meas[3]==1) x q[4];
if(meas[2]==1) z q[4];

cx q[4],q[5];
h q[4];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
if(meas[5]==1) x q[6];
if(meas[4]==1) z q[6];

cx q[6],q[7];
h q[6];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
if(meas[7]==1) x q[8];
if(meas[6]==1) z q[8];

cx q[8],q[9];
h q[8];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
if(meas[9]==1) x q[10];
if(meas[8]==1) z q[10];

cx q[10],q[11];
h q[10];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
if(meas[11]==1) x q[12];
if(meas[10]==1) z q[12];

cx q[12],q[13];
h q[12];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
if(meas[13]==1) x q[14];
if(meas[12]==1) z q[14];

measure q[14] -> meas[14];
