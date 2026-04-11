OPENQASM 2.0;
include "qelib1.inc";
qreg pos[8];
qreg coin[4];
creg meas_pos[8];
creg meas_coin[4];

x pos[4];

h coin[0];
h coin[1];
h coin[2];
h coin[3];

h coin[0];
h coin[1];
h coin[2];
h coin[3];

cx coin[0],pos[0];
cx coin[0],pos[1];
cx coin[1],pos[2];
cx coin[1],pos[3];
cx coin[2],pos[4];
cx coin[2],pos[5];
cx coin[3],pos[6];
cx coin[3],pos[7];

h coin[0];
h coin[1];
h coin[2];
h coin[3];

x coin[0];
x coin[1];
x coin[2];
x coin[3];
h coin[3];
ccx coin[0],coin[1],coin[2];
ccx coin[2],coin[3],coin[2];
ccx coin[0],coin[1],coin[2];
h coin[3];
x coin[0];
x coin[1];
x coin[2];
x coin[3];

cx coin[0],pos[1];
cx coin[0],pos[2];
cx coin[1],pos[3];
cx coin[1],pos[4];
cx coin[2],pos[5];
cx coin[2],pos[6];
cx coin[3],pos[0];
cx coin[3],pos[7];

h coin[0];
h coin[1];
h coin[2];
h coin[3];

x coin[0];
x coin[1];
x coin[2];
x coin[3];
h coin[3];
ccx coin[0],coin[1],coin[2];
ccx coin[2],coin[3],coin[2];
ccx coin[0],coin[1],coin[2];
h coin[3];
x coin[0];
x coin[1];
x coin[2];
x coin[3];

cx coin[0],pos[2];
cx coin[0],pos[3];
cx coin[1],pos[4];
cx coin[1],pos[5];
cx coin[2],pos[6];
cx coin[2],pos[7];
cx coin[3],pos[0];
cx coin[3],pos[1];

barrier pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],pos[7],coin[0],coin[1],coin[2],coin[3];
measure pos[0] -> meas_pos[0];
measure pos[1] -> meas_pos[1];
measure pos[2] -> meas_pos[2];
measure pos[3] -> meas_pos[3];
measure pos[4] -> meas_pos[4];
measure pos[5] -> meas_pos[5];
measure pos[6] -> meas_pos[6];
measure pos[7] -> meas_pos[7];
measure coin[0] -> meas_coin[0];
measure coin[1] -> meas_coin[1];
measure coin[2] -> meas_coin[2];
measure coin[3] -> meas_coin[3];
