# Step 1: Sample dataset (you can replace this with your own data)
x = [1, 2, 3, 4, 5]  # Input features
y = [2, 4, 5, 4, 5]  # Corresponding target values

# Step 2: Calculate mean of x and y
x̅ = sum(x) / len(x)
y̅ = sum(y) / len(y)

# Step 3: Calculate slope (m)
numerator = sum((xi - x̅) * (yi - y̅) for xi, yi in zip(x, y))
denominator = sum((xi - x̅) ** 2 for xi in x)
m = numerator / denominator

# Step 4: Calculate intercept (c)
c = y̅ - m * x̅

# Step 5: Make predictions
def predict(x_val):
    return c + m * x_val

# Example usage:
new_x = 6
predicted_y = predict(new_x)
print(f"Predicted y for x = {new_x}: {predicted_y:.2f}")
