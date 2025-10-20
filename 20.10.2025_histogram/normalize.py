import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def histogram_equalization(image_path, method=1):
    """
    Изравняване на хистограма за подобряване на контраста
    
    Args:
        image_path: Път към входното изображение
        method: Метод за изравняване (1, 2 или 3)
    
    Returns:
        Обработеното изображение като numpy array
    """
    
    # Зареждане на изображението
    img = Image.open(image_path).convert('L')  # Конвертиране в grayscale
    pixels = np.array(img)
    
    # Параметри
    Po = 0  # Начален пиксел
    Pmax = pixels.size - 1  # Последен пиксел
    Co = 0  # Начална стойност на пиксел
    Cmax = 255  # Максимална стойност на пиксел
    
    print(f"Размер на изображението: {pixels.shape}")
    print(f"Брой пиксели: {pixels.size}")
    
    # Стъпка 1: Изчисляване на хистограмата H(C)
    H = np.zeros(256, dtype=int)  # Хистограма за стойности 0-255
    
    for p in range(Po, Pmax + 1):
        pixel_value = pixels.flat[p]
        H[pixel_value] += 1
    
    # Изчисляване на средната стойност на хистограмата
    Hmid = pixels.size / 256  # Средна стойност на хистограмата
    
    print(f"Средна стойност на хистограмата (Hmid): {Hmid}")
    
    # Стъпка 2: Инициализация
    Rver = 0
    Hsum = 0
    L = np.zeros(256, dtype=int)  # L(C) - долна граница
    R = np.zeros(256, dtype=int)  # R(C) - горна граница
    Cn = np.zeros(256, dtype=int)  # Cn(C) - нова стойност
    
    # Стъпка 3: Изравняване на хистограмата
    for C in range(Co, Cmax + 1):
        L[C] = Rver
        Hsum += H[C]
        
        # Докато Hsum > Hmid, увеличаваме Rver
        while Hsum > Hmid:
            Hsum -= Hmid
            Rver += 1
        
        R[C] = Rver
        
        # Избор на метод за изчисляване на Cn(C)
        if method == 1:
            # Метод 1: Средна стойност
            Cn[C] = (L[C] + R[C]) // 2
        elif method == 2:
            # Метод 2: Случайна стойност в диапазона
            if L[C] <= R[C]:
                Cn[C] = random.randint(L[C], R[C])
            else:
                Cn[C] = L[C]
    
    # Стъпка 4: Прилагане на трансформацията към пикселите
    result_pixels = pixels.copy()
    
    for p in range(Po, Pmax + 1):
        original_value = pixels.flat[p]
        
        if method in [1, 2]:
            result_pixels.flat[p] = Cn[original_value]
        elif method == 3:
            # Метод 3: Използване на средната стойност
            result_pixels.flat[p] = Cn[original_value]
    
    return result_pixels, H

def main():
    """
    Главна функция за демонстрация на алгоритъма
    """
    print("=== Алгоритъм за изравняване на хистограма ===")
    print("Методи:")
    print("1 - Средна стойност")
    print("2 - Случайна стойност")
    
    # Създаване на тестово изображение (ако няма такова)
    # За демонстрация ще създадем просто изображение с ниски контрасти
    test_image = np.random.randint(50, 150, (100, 100), dtype=np.uint8)
    
    # Запазване на тестовото изображение
    Image.fromarray(test_image).save('test_input.png')
    print("Създадено тестово изображение: test_input.png")
    
    # Прилагане на алгоритъма
    method = 1  # Използваме метод 1
    processed_img, histogram_processed = histogram_equalization('test_input.png', method)
    
    # Запазване на резултата
    Image.fromarray(processed_img).save('test_output.png')
    print("Запазен резултат: test_output.png")
    
    # Изчисляване на хистограмата на оригиналното изображение
    histogram_original = np.zeros(256, dtype=int)
    for pixel in test_image.flat:
        histogram_original[pixel] += 1
    
    
    print("\n=== Статистики ===")
    print(f"Оригинал - мин: {test_image.min()}, макс: {test_image.max()}, средно: {test_image.mean():.2f}")
    print(f"Обработено - мин: {processed_img.min()}, макс: {processed_img.max()}, средно: {processed_img.mean():.2f}")

if __name__ == "__main__":
    main()
