
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];


rz(0.212) q[0];
rz(0.312) q[1];
rz(0.182) q[2];
cx q[0],q[1];
rz(0.156) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.221) q[2];
cx q[1],q[2];
cx q[0],q[2];
rz(0.193) q[2];
cx q[0],q[2];

rz(0.248) q[3];
rz(0.176) q[4];
rz(0.327) q[5];
cx q[3],q[4];
rz(0.112) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.267) q[5];
cx q[4],q[5];
cx q[3],q[5];
rz(0.231) q[5];
cx q[3],q[5];

rz(0.188) q[6];
rz(0.342) q[7];
rz(0.155) q[8];
cx q[6],q[7];
rz(0.245) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.198) q[8];
cx q[7],q[8];
cx q[6],q[8];
rz(0.273) q[8];
cx q[6],q[8];

rz(0.309) q[9];
rz(0.143) q[10];
rz(0.256) q[11];
cx q[9],q[10];
rz(0.189) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.217) q[11];
cx q[10],q[11];
cx q[9],q[11];
rz(0.274) q[11];
cx q[9],q[11];

rz(0.223) q[12];
rz(0.291) q[13];
rz(0.187) q[14];
cx q[12],q[13];
rz(0.241) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.178) q[14];
cx q[13],q[14];
cx q[12],q[14];
rz(0.235) q[14];
cx q[12],q[14];

cx q[0],q[3];
rz(0.162) q[3];
cx q[0],q[3];

cx q[3],q[6];
rz(0.213) q[6];
cx q[3],q[6];

cx q[6],q[9];
rz(0.174) q[9];
cx q[6],q[9];

cx q[9],q[12];
rz(0.229) q[12];
cx q[9],q[12];

cx q[1],q[4];
rz(0.194) q[4];
cx q[1],q[4];

cx q[4],q[7];
rz(0.251) q[7];
cx q[4],q[7];

cx q[7],q[10];
rz(0.168) q[10];
cx q[7],q[10];

cx q[10],q[13];
rz(0.232) q[13];
cx q[10],q[13];

cx q[2],q[5];
rz(0.204) q[5];
cx q[2],q[5];

cx q[5],q[8];
rz(0.183) q[8];
cx q[5],q[8];

cx q[8],q[11];
rz(0.215) q[11];
cx q[8],q[11];

cx q[11],q[14];
rz(0.257) q[14];
cx q[11],q[14];

rx(0.75) q[0];
rx(0.75) q[1];
rx(0.75) q[2];
rx(0.75) q[3];
rx(0.75) q[4];
rx(0.75) q[5];
rx(0.75) q[6];
rx(0.75) q[7];
rx(0.75) q[8];
rx(0.75) q[9];
rx(0.75) q[10];
rx(0.75) q[11];
rx(0.75) q[12];
rx(0.75) q[13];
rx(0.75) q[14];

barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
