# Dynamic-adaptive-MQKA

* Running **DA-MQKA.py** to obtain the simulation results.<br>

* The detailed simulation results using ***Qiskit*** and ***IBM Quantum Platform*** have been uploaded.<br>

## ***Multi-party QKA:***

1. Using the ***three-particle GHZ-like states*** to perform MQKA, when ***n=6***, ***m=5***, and the number of decoy states inserted is ***4***, the ***6*** participants ultimately obtain the same key ***KP = K1 || K2 || ... || Kn*** with length of ***60 bits***.<br>

2. Using the ***four-particle GHZ-like states*** to perform MQKA, when ***n=7***, ***m=6***, and the number of decoy states inserted is ***5***, the ***7*** participants ultimately obtain the same key ***KP = K1 || K2 || ... || Kn*** with length of ***84 bits***.<br>

## ***Dynamic Adaptive Key Update Mechanism:***

After the multi-party key agreement is completed, the simulation system will randomly test three dynamic scenarios:<br> 
1. Key update based on hash commitment when some participants leave; <br>
2. Sub-negotiation and key matching when new members join; <br>
3. Key update in a composite scenario where exit and joining occur simultaneously.<br>
