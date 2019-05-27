from pessoa import Pessoa

class Medico(Pessoa):
    def __init__(self):
        Pessoa.__init__(self)
        self._CRM = None
        self._telefone = None

    @property
    def CRM(self):
        return self._CRM

    @CRM.setter
    def CRM(self, CRM):
        self._CRM = CRM

    @property
    def telefone(self):
        return self._telefone

    @telefone.setter
    def telefone(self, telefone):
        self._telefone = telefone




