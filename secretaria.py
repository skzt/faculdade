from pessoa import Pessoa

class Secretaria(Pessoa):
    def __init__(self):
        Pessoa.__init__(self)
        self._matricula = None
        self._email = None

    @property
    def matricula(self):
        return self._matricula

    @matricula.setter
    def matricula(self, matricula):
        self._matricula = matricula

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, email):
        self._email = email






