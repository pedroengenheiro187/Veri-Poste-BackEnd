import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageChops, ImageStat
import io

def read_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image)

def compare_images(img1_array, img2_array):
    # Redimensionar imagens para o mesmo tamanho
    img1 = cv2.resize(img1_array, (500, 500))
    img2 = cv2.resize(img2_array, (500, 500))

    # Converter para escala de cinza
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Similaridade estrutural (SSIM)
    ssim_score, diff = ssim(gray1, gray2, full=True)
    ssim_percent = round(ssim_score * 100, 2)

    # Comparação de histograma de cores
    hist_diff = compare_histograms(img1, img2)

    # Verificação de rotação (por diferença de bordas)
    rotation_diff = detect_rotation(img1, img2)

    # Detecção simples de edição via ruído
    edited_score = detect_edits(img2)

    # Análise final
    analysis = {
        "similaridade_estrutura": f"{ssim_percent}%",
        "diferenca_histograma": f"{hist_diff}%",
        "possivel_rotacao": rotation_diff,
        "chance_de_edicao": f"{edited_score}%",
        "relatorio": gerar_relatorio(ssim_percent, hist_diff, rotation_diff, edited_score)
    }

    return analysis

def compare_histograms(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return round((1 - score) * 100, 2)

def detect_rotation(img1, img2):
    # Diferencia as bordas para tentar detectar rotação
    edges1 = cv2.Canny(img1, 100, 200)
    edges2 = cv2.Canny(img2, 100, 200)
    diff = np.mean(np.abs(edges1 - edges2))
    if diff > 15:
        return "Sim (ângulo ou rotação detectado)"
    else:
        return "Não detectado"

def detect_edits(img):
    # Detecção de edição via inconsistência de ruído
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    if variance < 80:
        return 80  # Alta chance de edição
    elif variance < 150:
        return 50
    else:
        return 10

def gerar_relatorio(sim, hist_diff, rotation, edit_chance):
    mensagens = []

    if sim > 90 and edit_chance < 30:
        mensagens.append("As imagens são quase idênticas e provavelmente não editadas.")
    elif sim > 70 and edit_chance < 50:
        mensagens.append("Imagens similares com pequenas variações, sem sinais fortes de edição.")
    elif sim < 50:
        mensagens.append("Imagens diferentes ou fortemente modificadas.")
    
    if hist_diff > 30:
        mensagens.append("Diferenças significativas de cor identificadas.")
    
    if rotation != "Não detectado":
        mensagens.append("Alteração de ângulo ou rotação detectada.")

    if edit_chance > 60:
        mensagens.append("Alta probabilidade de que a imagem tenha sido editada.")

    return " ".join(mensagens)
