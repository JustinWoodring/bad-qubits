u2(0,0) q[0];
u2(-pi,-pi) q[1];
cx q[0],q[1];
u2(-pi,-pi) q[0];
barrier q[0],q[1];
measure q[0] -> c[0];
