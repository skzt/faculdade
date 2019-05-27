from pessoa import Pessoa


class Paciente(Pessoa):
    def __init__(self):
        Pessoa.__init__(self)
        self._convenio = None
        self._telefone = None

    @property
    def convenio(self):
        return self._convenio

    @convenio.setter
    def convenio(self, convenio):
        self._convenio = convenio

    @property
    def telefone(self):
        return self._telefone

    @telefone.setter
    def telefone(self, telefone):
        self._telefone = telefone















