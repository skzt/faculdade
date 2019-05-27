from paciente import Paciente
from secretaria import Secretaria
from medico import Medico
from agenda import Agenda

paciente = Paciente()
secretaria = Secretaria()
medico = Medico()

paciente.cadastrarPessoa()
print(f"Paciente")
print(f"CPF: {paciente.CPF}")
print(f"Convenio: {paciente.convenio}")

medico.cadastrarPessoa()
print(f"\n\nSecretaria")
print(f"CPF: {secretaria.CPF}")
print(f"Matricula: {secretaria.matricula}")

medico.cadastrarPessoa()
print(f"\n\nMedico")
print(f"CPF: {medico.CPF}")
print(f"CRM: {medico.CRM}")

agenda = Agenda(paciente.CPF, '22/5/2019', '9:00', medico.CPF)
