
# VeriPoste Backend

API em Flask para comparar imagens e detectar edições ou similaridades com foco em obras de energia.

## Como usar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Rode a API:
```bash
python app.py
```

3. Faça uma requisição POST para `/analyze` com os campos `image1` e `image2` (form-data).

A resposta conterá similaridade de hash, SSIM, possíveis edições e conclusão.

---

Ideal para verificar fraudes em relatórios fotográficos.
