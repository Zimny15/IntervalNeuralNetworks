#Definicja klasy interval

class interval:
    def __init__(self, a, b=None):
        if b is None:
            #Punktowy przedział
            self.inf = a
            self.sup = a
        else:
            #Przedział ogólny
            if a > b:
                raise Exception("Lewa strona przedzialu nie moze byc wieksza od prawej!")
            self.inf = a
            self.sup = b

    def mid(self):
        return (self.sup + self.inf)/2
    def inv(self):
        a = self.inf
        b = self.sup
        if a > 0 or b < 0:
            return interval(1 / b, 1 / a)
        elif b == 0:
            return interval(1 / a, float('inf'))
        elif a == 0:
            return interval(float('-inf'), 1 / b)
        else:
            raise Exception("Nie mozna odwrocic przedzialu zawierajacego zero!")
    def find(self, x):
        if self.inf <= x <= self.sup:
            return True
        else: return False
        
    #Operacje arytmetyczne 
    def __add__(self, other):
        return interval(self.inf + other.inf, self.sup + other.sup)
    def __sub__(self, other):
        return interval(self.inf - other.sup, self.sup - other.inf)
    def __mul__(self, other):
        a = self.inf
        b = self.sup
        c = other.inf
        d = other.sup
        products = [a * c, a * d, b * c, b * d]
        return interval(min(products), max(products))
    def __truediv__(self, other):
        return self * other.inv()
    def __pow__(self, exp):
        a = self.inf
        b = self.sup
        if exp == 0:
            return interval(1,1)
        if exp % 2 == 0:
            min_val = min(abs(a), abs(b))**exp
            max_val = max(abs(a), abs(b))**exp
            if self.find(0):
                min_val = 0
            return interval(min_val, max_val)
        else: return interval(a**exp, b**exp)
    

    def __str__(self):
        return "[" + str(self.inf) + "," + str(self.sup) + "]"




    