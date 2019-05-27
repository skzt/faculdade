import os
from calendar import Calendar
from time import strftime


class Agenda:
    def __init__(self, paciente, data, horario, medico):
        self._paciente = paciente
        self._data = data
        self._horario = horario
        self._medico = medico
        self._status = False

        self._caminhoArquivo = os.path.join("./Arquivos", "Agendas", f"{str(self._medico)}.txt")

        # '*' significa que não há consulta marcada neste horario
        horarios = {'9:00': '*',
                    '10:00': '*',
                    '11:00': '*',
                    '14:00': '*',
                    '15:00': '*'}

        calendario = Calendar()

        with open(self._caminhoArquivo, 'a') as arquivo:
            hoje = (strftime('%d'), strftime('%m'), strftime('%Y'))
            arquivo.write(f"Ano: {int(hoje[2])}\n")

            for numeroMes in range(int(hoje[1]), 13):
                mes = calendario.itermonthdays2(int(hoje[2]), numeroMes)
                arquivo.write(f"Mês: {numeroMes}\n")

                for dia in mes:
                    if dia[0] == 0 or (numeroMes == int(hoje[1]) and dia[0] <= int(hoje[0])):
                        # Elimina os dias zerados.
                        # Elimina os dias que já passaram.
                        continue

                    arquivo.write(f"Dia: {dia[0]}-{dia[1]}-*-*-*-*-*\n")

    @property
    def paciente(self):
        return self._paciente

    @paciente.setter
    def paciente(self, paciente):
        self._paciente = paciente

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def horario(self):
        return self._horario

    @horario.setter
    def horario(self, horario):
        self._horario = horario

    @property
    def medico(self):
        return self._medico

    @medico.setter
    def medico(self, medico):
        self._medico = medico

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        assert isinstance(status, bool), f"Tipo invalido. Tipo esperado {bool}," \
            f" tipo encontrado {type(status)}"
        self._status = status
