# Math solutions that aren't defined in numpy or scipy

def derivative(coefficients):
    '''
    compute the derivative for a list of coefficients given in the following order:
      w0 + w1*x + w2*x^2 + w3*x^3 + ...
    where w0 is the coefficient with index 0, etc.
    '''
    derivative_coefficients = [index * c for index, c in enumerate(coefficients)][1:]
    
    return derivative_coefficients

def polyn_value(coefficients, x):
    '''
    compute the value of a polynomial expression for a list of coefficients given in the following order:
      w0 + w1*x + w2*x^2 + w3*x^3 + ...
    where w0 is the coefficient with index 0, etc.
    '''
    value = 0
    for index, c in enumerate(coefficients):
        value += c * x**index
    
    return value

 def linef2pts(pt1, pt2, solve="y"):
   '''
   Given two points (pt1 and pt2), return a function for the line
   between those two points
   
   expects pt1 and pt2 to each have length 2, dtype float
   default is to solve for y (returning y=f(x)); but you can also
   choose solve="x" to return x=f^-1(y)
   '''
   x1, y1 = pt1[0], pt1[1]
   x2, y2 = pt2[0], pt2[1]
 
   m = (y1 - y2) / (x1 - x2)
 
   if solve == "y":
     return lambda x: m * (x - x1) + y1
   elif solve == "x":
     return lambda y: (y - y1) / m + x1
   else:
     print "Can only solve for x or y"
 
 def linzero(fm1, a, b):
   '''
   Returns fm1(0) if the result is between a and b, None otherwise
   
   fm1 is assumed to be the inverse (f^-1(x)) function of f
   '''
   result = fm1(0)
 
   if result > b or result < a:
     result = None
 
   return result
