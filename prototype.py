#!/usr/bin/env python3
import argparse
import os
import base64
import io
from pdf2image import convert_from_path
from openai import OpenAI

# --- 関数定義 ---


def extract_text_via_gpt4o(pdf_path, client):
    """
    pdf2image を用いて PDF の各ページを画像化し、
    各画像を GPT-4o に渡してテキスト（数学的記述など）を抽出する。
    """
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, image in enumerate(images):
        print(f"[INFO] ページ {i+1} を GPT-4o で処理中...")
        page_text = process_image_with_gpt4o(image, client)
        full_text += page_text + "\n"
    return full_text


def process_image_with_gpt4o(image, client):
    """
    PIL Image オブジェクト image を GPT-4o に入力し、
    画像中の数学的記述や定理の要素を抽出する。

    ユーザー提示のサンプルコードに沿い、画像を base64 エンコードした data URL として送信します。
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    image_data = buf.getvalue()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:image/png;base64,{image_base64}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "以下の画像に含まれる数学的な記述や定理の要素を抽出してください。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def extract_main_theorem(full_text, client):
    """
    論文全文から、主定理とその中で用いられている主要な用語の定義を抽出するため、
    GPT-4o にプロンプトを送信する。
    """
    prompt = (
        "以下の数学論文の本文から、主定理とその中で用いられている主要な用語の定義を抽出してください。\n\n"
        "【論文本文】\n" + full_text + "\n\n"
        "抽出結果は、主定理の記述と、用語一覧・その定義の形式で示してください。"
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def prove_theorem(main_theorem_text, client):
    """
    抽出した主定理に対して、GPT-4o を用いて詳細な証明を生成する。
    """
    prompt = (
        "以下の定理について、詳細な証明を示してください。\n\n"
        "【定理】\n" + main_theorem_text
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def validate_proof(main_theorem_text, proof_text, client):
    """
    生成された証明が概要レベルで正しいか、また主要な論点が網羅されているかを評価するため、
    GPT-4o に検証を依頼する。
    """
    prompt = (
        "以下の定理とその証明について、概要レベルで正しいかどうか、また主要な論点が網羅されているか評価してください。\n\n"
        "【定理】\n" + main_theorem_text + "\n\n"
        "【証明】\n" + proof_text + "\n\n"
        "評価とともに、必要ならば改善点も指摘してください。"
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def save_result(output_dir, filename, text):
    """指定したディレクトリにファイルとして結果を保存する"""
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] {filename} に結果を保存しました。")


def load_result(output_dir, filename):
    """指定したディレクトリから結果ファイルを読み込む"""
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "r", encoding="utf-8") as f:
        return f.read()

# --- メイン処理 ---


def main():
    parser = argparse.ArgumentParser(
        description="PDF から数学論文の主定理抽出・証明生成・検証を行い、途中結果をファイルに保存する CLI ツール"
    )
    parser.add_argument("pdf_path", help="ローカルに保存済みの PDF ファイルのパス")
    parser.add_argument("-o", "--output-dir",
                        required=True, help="結果の出力先ディレクトリ")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] 出力ディレクトリ '{output_dir}' を作成しました。")

    client = OpenAI()

    # --- 1. PDF からのテキスト抽出 ---
    extracted_text_file = "extracted_text.txt"
    if os.path.exists(os.path.join(output_dir, extracted_text_file)):
        print("[INFO] 既存の抽出テキストファイルを読み込みます...")
        extracted_text = load_result(output_dir, extracted_text_file)
    else:
        print("[INFO] PDF から GPT-4o を用いてテキスト抽出を実施します...")
        extracted_text = extract_text_via_gpt4o(args.pdf_path, client)
        save_result(output_dir, extracted_text_file, extracted_text)

    # --- 2. 主定理と用語の抽出 ---
    main_theorem_file = "main_theorem.txt"
    if os.path.exists(os.path.join(output_dir, main_theorem_file)):
        print("[INFO] 既存の主定理ファイルを読み込みます...")
        main_theorem = load_result(output_dir, main_theorem_file)
    else:
        print("[INFO] GPT-4o を用いて主定理と用語の抽出を実施します...")
        main_theorem = extract_main_theorem(extracted_text, client)
        save_result(output_dir, main_theorem_file, main_theorem)

    # --- 3. 定理の証明生成 ---
    proof_file = "proof.txt"
    if os.path.exists(os.path.join(output_dir, proof_file)):
        print("[INFO] 既存の証明ファイルを読み込みます...")
        proof = load_result(output_dir, proof_file)
    else:
        print("[INFO] GPT-4o を用いて証明生成を実施します...")
        proof = prove_theorem(main_theorem, client)
        save_result(output_dir, proof_file, proof)

    # --- 4. 証明の概要検証 ---
    validation_file = "validation.txt"
    if os.path.exists(os.path.join(output_dir, validation_file)):
        print("[INFO] 既存の検証結果ファイルを読み込みます...")
        validation = load_result(output_dir, validation_file)
    else:
        print("[INFO] GPT-4o を用いて証明の検証を実施します...")
        validation = validate_proof(main_theorem, proof, client)
        save_result(output_dir, validation_file, validation)

    print("[INFO] すべての処理が完了しました。")
    print("----- 抽出テキスト -----")
    print(extracted_text)
    print("----- 主定理と用語 -----")
    print(main_theorem)
    print("----- 証明 -----")
    print(proof)
    print("----- 検証結果 -----")
    print(validation)


if __name__ == "__main__":
    main()
