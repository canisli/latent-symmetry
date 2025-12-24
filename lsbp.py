def solve(s: str):
    stack = []
    count = 0
    for char in s:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
                count += 2
    return count

print(solve('(()())'))