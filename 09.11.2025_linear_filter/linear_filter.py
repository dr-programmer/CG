import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_linear_filter(image, kernel):
    """Прилага линейна филтрация (конволюция) върху изображението"""
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered

def main():
    path = input("Въведете път до изображението (или натиснете Enter за пример): ")
    if not path:
        size = 512
        square_size = 64
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Създаваме шахова дъска с по-големи квадрати
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
        
        # Добавяме някои кръгове за по-добра видимост на ефектите на филтъра
        center = size // 2
        cv2.circle(image, (center, center), 100, 128, 3)
        cv2.circle(image, (center, center), 150, 200, 5)
        
        # Добавяме някои линии
        cv2.line(image, (0, size//4), (size, size//4), 180, 3)
        cv2.line(image, (0, 3*size//4), (size, 3*size//4), 180, 3)
        cv2.line(image, (size//4, 0), (size//4, size), 180, 3)
        cv2.line(image, (3*size//4, 0), (3*size//4, size), 180, 3)
        
        cv2.imwrite('image.png', image)
        print("Създадено тестово изображение: image.png")
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Грешен път или неподдържан формат на изображението!")

    print(f"Размер на изображението: {image.shape}")

    print("\nПример: 3x3 филтър за изостряне:")
    print("0 -1  0\n-1  5 -1\n0 -1  0\n")
    print("Можете да въведете собствен филтър (разделяйте с интервали и нов редове).")
    print("Натиснете Enter за примерния филтър.")

    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append([float(x) for x in line.split()])

    if lines:
        kernel = np.array(lines, dtype=np.float32)
    else:
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)

    kernel /= np.sum(kernel) if np.sum(kernel) != 0 else 1

    print("\nФилтърът е:\n", kernel)

    filtered = apply_linear_filter(image, kernel)

    cv2.imwrite('filtered.png', filtered)
    print("Запазен резултат: filtered.png")

if __name__ == "__main__":
    main()
