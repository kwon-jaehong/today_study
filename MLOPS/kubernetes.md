패캠
-----------

명령어

kubectl api-resources
- 쿠버네티스 클러스터에서 사용할 수 있는 오브젝트 목록 조회

kubectl explain <type>
- 쿠버네티스 오브젝트의 설명과 1레벨 속성들의 설명
- apiversion, kind, metadata, spec, status

kubectl explain <type>.<fieldName>
- kubectl explain pods.spec.containers 등
- 쿠버네티스 오브젝트 속성들의 구체적인 설명
