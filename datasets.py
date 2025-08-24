import random
from typing import List, Optional

def real_positive_decimals(n_samples: int, decimal_places: int, random_state: Optional[int] = None) -> List[str]:
    """
    Generate random decimal strings from 0 to 1 with specified decimal places.
    Args:
        n_samples: Number of samples to generate
        decimal_places: Number of decimal places (precision)
        random_state: Random seed for reproducible results
    Returns:
        List of decimal strings between 0 and 1
    """
    if random_state is not None:
        random.seed(random_state)
    
    decimals = []
    for i in range(n_samples):
        # Generate integer part (always 0 for 0-1 range)
        int_part = "0"
        
        # Generate fractional part with exact precision
        frac_part = ""
        for j in range(decimal_places):
            frac_part += str(random.randint(0, 9))
        
        decimal_str = f"{int_part}.{frac_part}"
        decimals.append(decimal_str)
    return decimals

def real_positive_and_negative_decimals(n_samples: int, decimal_places: int, random_state: Optional[int] = None) -> List[str]:
    """
    Generate random decimal strings from -1 to 1 with specified decimal places.
    Args:
        n_samples: Number of samples to generate
        decimal_places: Number of decimal places (precision)
        random_state: Random seed for reproducible results
    Returns:
        List of decimal strings between -1 and 1
    """
    if random_state is not None:
        random.seed(random_state)
    
    decimals = []
    for i in range(n_samples):
        # Generate sign
        sign = random.choice(["-", ""])
        
        # Generate integer part (0 for -1 to 1 range)
        int_part = "0"
        
        # Generate fractional part with exact precision
        frac_part = ""
        for j in range(decimal_places):
            frac_part += str(random.randint(0, 9))
        
        decimal_str = f"{sign}{int_part}.{frac_part}"
        decimals.append(decimal_str)
    return decimals

def real_int_and_decimal(n_samples: int,
                        digits_before: int,
                        digits_after: int,
                        random_state: Optional[int] = None) -> List[str]:
    """
    Generate random decimal strings with specified digits before and after decimal point.
    Args:
        n_samples: Number of samples to generate
        digits_before: Number of digits before decimal point
        digits_after: Number of digits after decimal point
        random_state: Random seed for reproducible results
    Returns:
        List of decimal strings with specified precision
    """
    if random_state is not None:
        random.seed(random_state)
    
    decimals = []
    
    for i in range(n_samples):
        # Generate sign
        sign = random.choice(["-", ""])
        
        # Generate integer part with exact number of digits
        int_part = ""
        # First digit can't be 0 (unless digits_before is 1)
        if digits_before == 1:
            int_part = str(random.randint(0, 9))
        else:
            int_part = str(random.randint(1, 9))  # First digit 1-9
            for j in range(digits_before - 1):
                int_part += str(random.randint(0, 9))
        
        # Generate fractional part with exact precision
        frac_part = ""
        for j in range(digits_after):
            frac_part += str(random.randint(0, 9))
        
        if digits_after > 0:
            decimal_str = f"{sign}{int_part}.{frac_part}"
        else:
            decimal_str = f"{sign}{int_part}"
            
        decimals.append(decimal_str)
    return decimals

# Example usage and testing
if __name__ == "__main__":
    print("String-Based Decimal Dataset Generator")
    print("=" * 50)
    
    # Test reproducibility
    print("\n0. Testing reproducibility with random_state:")
    dataset_a1 = real_positive_decimals(5, 3, random_state=42)
    dataset_a2 = real_positive_decimals(5, 3, random_state=42)
    dataset_b = real_positive_decimals(5, 3, random_state=123)
    
    print(f"With seed 42 (run 1): {dataset_a1}")
    print(f"With seed 42 (run 2): {dataset_a2}")
    print(f"With seed 123:       {dataset_b}")
    print(f"Run 1 == Run 2: {dataset_a1 == dataset_a2}")
    print(f"Seed 42 == Seed 123: {dataset_a1 == dataset_b}")
    
    # Test dataset 1: 0 to 1
    print("\n1. Dataset 1: Random decimals from 0 to 1")
    dataset1 = real_positive_decimals(5, 4, random_state=42)
    print(f"Sample (4 decimal places): {dataset1}")
    
    # Test dataset 2: -1 to 1
    print("\n2. Dataset 2: Random decimals from -1 to 1")
    dataset2 = real_positive_and_negative_decimals(5, 4, random_state=42)
    print(f"Sample (4 decimal places): {dataset2}")
    
    # Test dataset 3: Extreme precision test
    print("\n3. Dataset 3: Extreme precision test")
    dataset3 = real_int_and_decimal(5, 12, 20, random_state=42)
    print(f"Sample (12 digits before, 20 after):")
    
    # Individual display with full precision
    print("\nIndividual values with TRUE 20 decimal places:")
    for i, d in enumerate(dataset3, 1):
        print(f"  {i}: {d}")
    
    # Test even more extreme precision
    print("\n4. Dataset 4: Ultra-extreme precision test")
    dataset4 = real_int_and_decimal(3, 5, 50, random_state=42)
    print(f"Sample (5 digits before, 50 after):")
    for i, d in enumerate(dataset4, 1):
        print(f"  {i}: {d}")