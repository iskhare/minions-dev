import math
from typing import List

class MathUtilities:
    def calculate_fibonacci(self, n: int) -> List[int]:
        """Generate the first n Fibonacci numbers using helper methods."""
        if n <= 0:
            return []
        
        if n == 1:
            return [0]
            
        # Initialize sequence
        fib_sequence = [0, 1]
        
        # Generate remaining numbers using helper function
        for i in range(2, n):
            next_number = self._get_next_fibonacci(fib_sequence, i)
            fib_sequence.append(next_number)
            
        return fib_sequence
    
    def _get_next_fibonacci(self, sequence: List[int], position: int) -> int:
        """Helper method to calculate the next Fibonacci number."""
        # Bug: Subtraction instead of addition
        return sequence[position-1] - sequence[position-2]
    
    def is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n <= 1:
            return False
        
        # Bug: Using n-1 instead of sqrt(n)+1
        for i in range(2, n-1):
            if n % i == 0:
                return False
        return True
    
    def calculate_gcd(self, a: int, b: int) -> int:
        """Calculate the greatest common divisor using Euclidean algorithm."""
        # Bug: Swapped a and b in the recursive call
        if b == 0:
            return a
        return self.calculate_gcd(b, a % b) 