import json


def load_law_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_content_by_chunk_id(data, chunk_id):

    for item in data:
        if item.get("chunk_id").lower() == chunk_id.lower():
            return item.get("content")

    return None


if __name__ == "__main__":
    json_path = "legal_traffic_chunks.json"


    data = load_law_data(json_path)

    print("===== TRA CỨU NỘI DUNG LUẬT =====")
    print("Nhập 'exit' hoặc 'quit' để thoát.\n")

    while True:
        chunk_id = input("Nhập chunk_id: ").strip()

        if chunk_id.lower() in ("exit", "quit"):
            print("Đã thoát chương trình.")
            break

        content = get_content_by_chunk_id(data, chunk_id)

        if content:
            print("\n===== Nội dung luật =====")
            print(content)
        else:
            print(f"Không tìm thấy chunk_id: {chunk_id}")

        print("-" * 60)