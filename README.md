# weapon_detection

사람을 탐지하고 실시간 위치를 출력
무기(망치)를 든 사람을 위험 인물(Dangerous Human)이라 인식하고 타겟팅
- 타겟팅된 위험 인물의 위치만 출력

[문제 및 개선사항]</br>
depth 카메라의 깊이(Z값)과 YOLO에서의 바운딩 박스의 중심점 좌표(X, Y값)를 통해 실제 위치를 출력해야 하는데
직선 거리는 출력하는데 용이하나 사선 거리는 피타고라스로 구하려고 적용해보았으나 실제 위치와의 오차가 매우 큼

추후 실제 위치를 반영할 수 있는 계산 식을 수정할 예정


[추가해야 할 내용]</br>
중심점 좌표와 인근 10픽셀의 깊이 값의 평균 값을 depth값으로 활용

#### 코드:
```python
def calculate_average_depth(center_x, center_y, depth_frame, window_size=3):
    depth_values = []
    offset = window_size // 2  # 윈도우 크기 절반

    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            sampled_depth = depth_frame.get_distance(center_x + dx, center_y + dy)
            if sampled_depth > 0:  # 유효한 깊이만 포함
'''
                depth_values.append(sampled_depth)
    
    if not depth_values:
        return None  # 유효 깊이가 없는 경우

    return sum(depth_values) / len(depth_values)  # 평균 깊이 반환
