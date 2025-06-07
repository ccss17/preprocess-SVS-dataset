import os

def add_marker_until_filename(input_path, output_path, target_filename):
    """
    텍스트 파일을 한 줄씩 읽어, 특정 파일 이름이 나오기 전까지의 모든 줄 끝에 '|O'를 추가합니다.

    Args:
        input_path (str): 원본 텍스트 파일 경로.
        output_path (str): 결과를 저장할 파일 경로.
        target_filename (str): 처리를 중단할 기준이 되는 파일 이름.
    """
    try:
        # 파일을 읽고 쓰기 위해 'utf-8' 인코딩으로 엽니다.
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            target_found = False  # 목표 파일 이름을 찾았는지 여부를 추적하는 플래그

            for line in infile:
                # 줄 끝의 공백(개행 문자 등)을 제거합니다.
                stripped_line = line.strip()

                # 빈 줄은 그대로 출력 파일에 씁니다.
                if not stripped_line:
                    outfile.write('\n')
                    continue

                # 아직 목표 파일 이름을 찾지 못했다면
                if not target_found:
                    # '|'를 기준으로 줄을 나누고 첫 번째 요소(파일 이름)를 가져옵니다.
                    current_filename = stripped_line.split('|')[0]

                    # 현재 줄의 파일 이름이 목표 파일 이름과 같은지 확인합니다.
                    if current_filename == target_filename:
                        target_found = True  # 플래그를 True로 설정
                        # 목표 줄은 원본 그대로 씁니다.
                        outfile.write(line)
                    else:
                        # 목표 파일 이름이 아니면, 줄 끝에 '|O'를 추가하고 개행 문자를 붙여 씁니다.
                        outfile.write(stripped_line + '|O\n')
                
                # 목표 파일 이름을 이미 찾았다면
                else:
                    # 원본 줄을 그대로 씁니다.
                    outfile.write(line)
                    
        print(f"처리 완료! 결과가 '{output_path}' 파일에 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")


# --- 이 부분을 수정하여 사용하세요 ---

preprocessed_gv_path =  "/home/ccss17/dataset/gv_dataset_preprocessed"
metadata_path = preprocessed_gv_path + "/metadata.txt"
metadata_path_v1 = preprocessed_gv_path + "/metadata_1.txt"

if __name__ == "__main__":
    # 1. 원본 파일 경로를 지정하세요.
    input_file_path = metadata_path

    # 2. 결과가 저장될 파일 경로를 지정하세요.
    #    원본과 다른 이름으로 지정하는 것을 권장합니다.
    output_file_path = metadata_path_v1

    # 3. 처리를 멈출 기준이 되는 파일 이름을 정확하게 입력하세요.
    #    (예: 'file_123.wav')
    target_filename_to_stop = 'SINGER_79_30TO49_CLEAR_FEMALE_BALLAD_C3432_21.wav'

    # 함수 실행
    add_marker_until_filename(input_file_path, output_file_path, target_filename_to_stop)
