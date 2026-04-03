import torch
import torch.nn as nn
# Импортируем правильные классы из библиотеки
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletDecomposition(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        """
        Модуль для прямого дискретного вейвлет-преобразования (DWT).
        - J: количество уровней декомпозиции.
        - wave: тип вейвлета (например, 'db1', 'haar').
        - mode: режим расширения сигнала на границах.
        """
        super().__init__()
        # Инициализируем предобученный модуль DWTForward
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)

    def forward(self, x):
        """
        Вход: x — тензор изображения (B, C, H, W)
        Выход: 
        - yl: низкочастотная (аппроксимирующая) составляющая.
        - yh: список высокочастотных (детализирующих) составляющих.
              Каждый элемент yh имеет форму (B, C, 3, H', W'), где 3 — это каналы LH, HL, HH.
        """
        yl, yh = self.dwt(x)
        # Извлекаем три компонента из первого (и единственного) уровня декомпозиции
        # Они уже представлены как единый тензор размерности 3
        lh, hl, hh = yh[0][:, :, 0, :, :], yh[0][:, :, 1, :, :], yh[0][:, :, 2, :, :]
        return yl, lh, hl, hh


class WaveletReconstruction(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        """
        Модуль для обратного дискретного вейвлет-преобразования (IDWT).
        """
        super().__init__()
        # Инициализируем предобученный модуль DWTInverse
        self.idwt = DWTInverse(wave=wave, mode=mode)

    def forward(self, yl, lh, hl, hh):
        """
        Вход: низкочастотная и три высокочастотные составляющие.
        Выход: восстановленное изображение.
        """
        # Объединяем три высокочастотные компоненты в один тензор размерности 3
        yh = torch.stack([lh, hl, hh], dim=2)
        # Выполняем обратное преобразование
        return self.idwt((yl, [yh]))