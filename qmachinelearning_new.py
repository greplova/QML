from projectq import MainEngine
from projectq.ops import *
import numpy as numpy
import scipy as scipy
import scipy.optimize as scipyopt

eng = MainEngine()


# The gates are defined as a class
# Here we define the new gate based on the class: BasicRotationGate
class NewGate(BasicRotationGate):
	# The first function of the class is initialization
	# Which will take two arguments: phi and pgk
	# See detials here: https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/projectq/ops/_basics.py
	def __init__(self, phi, pgk):
		BasicGate.__init__(self)
		self._angle = float(phi)
		self._pgk = float(pgk)

	# The gate will be defined by the matrix as for the Ph gate
	# https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/projectq/ops/_gates.py
	@property
	def matrix(self):
		pgkX = self._pgk
		pgkI = 1 - self._pgk
		return np.matrix([[np.sqrt(pgkI), np.sqrt(pgkX)*cmath.exp(1j * self._angle)],[np.sqrt(pgkX)*cmath.exp(-1j * self._angle), np.sqrt(pgkI)]])



def CNewGate(n,phi,pgk):
	return C(NewGate(phi,pgk),n)

def New_Circuit(phis,pgks,xinput):
	qubit1 = eng.allocate_qubit()
	qubit2 = eng.allocate_qubit()
	qubit3 = eng.allocate_qubit()
	qubit4 = eng.allocate_qubit()
	ancilla = eng.allocate_qubit()

	if numpy.mod(xinput,2) == 1:
		X | qubit1

	if numpy.mod(numpy.floor(xinput/2),2) == 1:
		X | qubit2

	if numpy.mod(numpy.floor(xinput/4),2) == 1:
		X | qubit3

	if numpy.mod(numpy.floor(xinput/8),2) == 1:
		X | qubit4


	print(phis)
	print(pgks)

	NewGate(phis[0], pgks[0]) | ancilla

	CNewGate(1,phis[1],pgks[1]) | (qubit1, ancilla)
	CNewGate(1,phis[2],pgks[2]) | (qubit2, ancilla)
	CNewGate(1,phis[3],pgks[3]) | (qubit3, ancilla)
	CNewGate(1,phis[4],pgks[4]) | (qubit4, ancilla)

	eng.flush()

	CNewGate(2,phis[5],pgks[5]) | (qubit2, qubit1, ancilla)
	eng.flush()
	CNewGate(2,phis[6],pgks[6]) | (qubit3, qubit1, ancilla)
	eng.flush()
	CNewGate(2,phis[7],pgks[7]) | (qubit3, qubit2, ancilla)
	eng.flush()

	CNewGate(3,phis[8],pgks[8]) | (qubit3, qubit2, qubit1, ancilla)
	eng.flush()
	CNewGate(2,phis[9],pgks[9]) | (qubit4, qubit1, ancilla)
	eng.flush()
	CNewGate(2,phis[10],pgks[10]) | (qubit4, qubit2, ancilla)
	eng.flush()




	CNewGate(3,phis[11],pgks[11]) | (qubit4, qubit2, qubit1, ancilla)
	eng.flush()
	CNewGate(2,phis[12],pgks[12]) | (qubit4, qubit3, ancilla)
	eng.flush()
	CNewGate(3,phis[13],pgks[13]) | (qubit4, qubit3, qubit1, ancilla)
	eng.flush()
	CNewGate(3,phis[14],pgks[14]) | (qubit4, qubit3, qubit2, ancilla)
	eng.flush()

	CNewGate(4,phis[15],pgks[15]) | (qubit4, qubit3, qubit2, qubit1, ancilla)
	eng.flush()

	prob0 = eng.backend.get_probability([0],ancilla)
	Measure | qubit1
	Measure | qubit2
	Measure | qubit3
	Measure | qubit4
	Measure | ancilla
	return prob0

def get_F(x,args=[]):
	phis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	pgks = x

	prob0_array = []
	for n in range(16):
		xinput = n
		prob0n = New_Circuit(phis,pgks,xinput)
		prob0_array.append(prob0n)
	prob0_array = numpy.asarray(prob0_array)
	F = np.prod(prob0_array)**(1/32)
	return 1-F


boundarray=[(0,1), (0, 1), (0, 1), (0, 1), (0, 1),(0,1), (0, 1), (0, 1), (0, 1), (0, 1), (0,1), (0, 1), (0, 1), (0, 1), (0, 1), (0,1)]
A=scipyopt.differential_evolution(get_F, boundarray,args=[],mutation=(0,1.8))
print A
