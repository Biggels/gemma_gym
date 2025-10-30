
def is_valid_isbn10(isbn):
    # Check if the string has exactly 10 characters
    if len(isbn) != 10:
        return False
    
    # Check if the first 9 characters are digits
    for i in range(9):
        if not isbn[i].isdigit():
            return False
    
    # Check if the last character is either a digit or 'X' (for 10)
    last_char = isbn[9]
    if last_char not in '0123456789X':
        return False
    
    # Calculate the checksum
    total = 0
    for i in range(10):
        digit = int(isbn[i]) if i < 9 else (10 if last_char == 'X' else int(last_char))
        total += digit * (10 - i)
    
    # Check if the total modulo 11 is 0
    return total % 11 == 0
