u2(0,-pi) eval[0];
u2(0,-pi) eval[1];
u2(0,-pi) eval[2];
u2(0,-pi) eval[3];
u2(0,-pi) eval[4];
u2(0,-pi) eval[5];
u2(0,-pi) eval[6];
u2(0,-pi) eval[7];
u3(0.9272952180016122,0,0) q[0];
u3(0.8553167381645228,0,0) q[1];
u3(1.1246290255627306,0,0) q[2];
u3(0.7721734635720992,0,0) q[3];
u3(1.0352761804100977,0,0) q[4];
u3(0.6124568952367110,0,0) q[5];
u3(1.2831853071795862,0,0) q[6];

cx eval[0],q[0];
u(-0.9272952180016122,0,0) q[0];
cx eval[0],q[0];

cx eval[1],q[1];
u(-0.8553167381645228,0,0) q[1];
cx eval[1],q[1];

cx eval[2],q[2];
u(-1.1246290255627306,0,0) q[2];
cx eval[2],q[2];

cx eval[3],q[3];
u(-0.7721734635720992,0,0) q[3];
cx eval[3],q[3];

cx eval[4],q[4];
u(-1.0352761804100977,0,0) q[4];
cx eval[4],q[4];

cx eval[5],q[5];
u(-0.6124568952367110,0,0) q[5];
cx eval[5],q[5];

cx eval[6],q[6];
u(-1.2831853071795862,0,0) q[6];
cx eval[6],q[6];

h eval[0];
h eval[1];
h eval[2];
h eval[3];
h eval[4];
h eval[5];
h eval[6];
h eval[7];

u(0.9272952180016122,0,0) q[0];
u(0.8553167381645228,0,0) q[1];
u(1.1246290255627306,0,0) q[2];
u(0.7721734635720992,0,0) q[3];
u(1.0352761804100977,0,0) q[4];
u(0.6124568952367110,0,0) q[5];
u(1.2831853071795862,0,0) q[6];

cp(-pi/2) eval[6],eval[7];
cp(-pi/4) eval[5],eval[7];
cp(-pi/8) eval[4],eval[7];
cp(-pi/16) eval[3],eval[7];
cp(-pi/32) eval[2],eval[7];
cp(-pi/64) eval[1],eval[7];
cp(-pi/128) eval[0],eval[7];

cp(-pi/2) eval[5],eval[6];
cp(-pi/4) eval[4],eval[6];
cp(-pi/8) eval[3],eval[6];
cp(-pi/16) eval[2],eval[6];
cp(-pi/32) eval[1],eval[6];
cp(-pi/64) eval[0],eval[6];

cp(-pi/2) eval[4],eval[5];
cp(-pi/4) eval[3],eval[5];
cp(-pi/8) eval[2],eval[5];
cp(-pi/16) eval[1],eval[5];
cp(-pi/32) eval[0],eval[5];

cp(-pi/2) eval[3],eval[4];
cp(-pi/4) eval[2],eval[4];
cp(-pi/8) eval[1],eval[4];
cp(-pi/16) eval[0],eval[4];

cp(-pi/2) eval[2],eval[3];
cp(-pi/4) eval[1],eval[3];
cp(-pi/8) eval[0],eval[3];

cp(-pi/2) eval[1],eval[2];
cp(-pi/4) eval[0],eval[2];

cp(-pi/2) eval[0],eval[1];

barrier eval[0],eval[1],eval[2],eval[3],eval[4],eval[5],eval[6],eval[7],q[0],q[1],q[2],q[3],q[4],q[5],q[6];

measure eval[0] -> meas[0];
measure eval[1] -> meas[1];
measure eval[2] -> meas[2];
measure eval[3] -> meas[3];
measure eval[4] -> meas[4];
measure eval[5] -> meas[5];
measure eval[6] -> meas[6];
measure eval[7] -> meas[7];
measure q[0] -> meas[8];
measure q[1] -> meas[9];
measure q[2] -> meas[10];
measure q[3] -> meas[11];
measure q[4] -> meas[12];
measure q[5] -> meas[13];
measure q[6] -> meas[14];
