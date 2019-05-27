class Pessoa:
    def __init__(self):
        self._CPF = None
        self._nome = None
        self._tipoPessoa = None

        self._caminhoArquivo = "./Arquivos/pessoas.txt"

        try:
            open(self._caminhoArquivo, 'x')
        except IOError as error:
            print(error)

    def _validarCPF(self):
        CPF = [int(digito) for digito in list(self._CPF)]
        soma = 0
        for numero, digito in zip(range(10, 1, -1), CPF):
            soma += (numero * digito)
        if ((soma * 10) % 11) % 10 != CPF[-2]:
            return False

        soma = 0
        for numero, digito in zip(range(11, 1, -1), CPF):
            soma += (numero * digito)
        if ((soma * 10) % 11) % 10 != CPF[-1]:
            return False

        return True

    def cadastrarPessoa(self):
        self._CPF = input("Digite o CPF: ")
        if self.existeCPF():
            print("Cliente já cadastrado.")
            # TODO: Manter na tela de cadastro de GUI
            exit()
        else:
            if self._validarCPF() is False:
                print("CPF Invalido")
                # TODO: Manter na tela de cadastro de GUI
                exit()
            else:
                self._nome = input("Digite o Nome:")
                while self._nome == '':
                    self._nome = input("Digite o Nome: ")

                self._tipoPessoa = input("Digite o tipo de Pessoa:")
                with open(self._caminhoArquivo, 'a') as arquivo:
                    arquivo.write(f"CPF: {self._CPF}\n"
                                  f"Nome: {self._nome}\n"
                                  f"tipoPessoa: {self.tipoPessoa}\n")

    def existeCPF(self):
        with open(self._caminhoArquivo, 'r') as arquivo:
            for linha in arquivo:
                if "CPF" in linha.rstrip().upper():
                    if self._CPF == linha.rstrip().split()[1]:
                        return True

        return False

    @property
    def CPF(self):
        return self._CPF

    @CPF.setter
    def CPF(self, CPF):
        # TODO: verificar se CPF é valido
        try:
            self._validarCPF(CPF)
        except AssertionError as error:
            print(error)
            exit()
        self._CPF = CPF

    @property
    def nome(self):
        return self._nome

    @nome.setter
    def nome(self, nome):
        self._nome = nome

    @property
    def tipoPessoa(self):
        return self._tipoPessoa

    @tipoPessoa.setter
    def tipoPessoa(self, tipoPessoa):
        self._tipoPessoa = tipoPessoa
