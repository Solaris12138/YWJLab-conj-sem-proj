NUM_TOPICS = 7
REJECT_SID = ["sub-20251120smy"]


NUM_VERTEX_DICT = {
    "ico4" : 5124,
    "ico5" : 20484,
}


ATLAS = "Schaefer2018_400Parcels_17Networks_order"
MidTL_LH = ["17Networks_LH_DefaultB_Temp_4-lh",
            "17Networks_LH_DefaultB_Temp_6-lh",
            "17Networks_LH_TempPar_3-lh",
            "17Networks_LH_TempPar_2-lh"]
MidTL_RH = ["17Networks_RH_TempPar_2-rh",
            "17Networks_RH_TempPar_3-rh",
            "17Networks_RH_TempPar_4-rh"]
ATL_LH = ["17Networks_LH_DefaultB_Temp_1-lh"]
ATL_RH = ["17Networks_RH_LimbicA_TempPole_3-rh"]


"""
Block Setting Example:
    Block 1 
        Block-1_topic-N_sent-1: "因为他努力学习，所以成绩很优秀" Sta-Caus
        Block-1_topic-N_sent-2: "因为他努力学习，所以成绩很糟糕" DevSem-Cons
        Block-1_topic-N_sent-3: "虽然他努力学习，但是成绩很优秀" DevSem-Caus
        Block-1_topic-N_sent-4: "虽然他努力学习，但是成绩很糟糕" Sta-Cons
    Block 2 
        Block-2_topic-N_sent-1: "因为他努力学习，所以成绩很优秀" Sta-Caus
        Block-2_topic-N_sent-2: "因为他努力学习，但是成绩很糟糕" DevSynC2-Cons
        Block-2_topic-N_sent-3: "虽然他努力学习，所以成绩很优秀" DevSynC2-Caus
        Block-2_topic-N_sent-4: "虽然他努力学习，但是成绩很糟糕" Sta-Cons
    Block 3 
        Block-3_topic-N_sent-1: "因为他努力学习，所以成绩很优秀" Sta-Caus
        Block-3_topic-N_sent-2: "因为他努力学习，但是成绩很优秀" DevSynC1-Caus
        Block-3_topic-N_sent-3: "虽然他努力学习，所以成绩很糟糕" DevSynC1-Cons
        Block-3_topic-N_sent-4: "虽然他努力学习，但是成绩很糟糕" Sta-Cons
"""

TOPIC_SEM_SETTINGS = {
    "topic1" : {
        "contexts" : ["因为他努力学习，所以成绩很", "虽然他努力学习，但是成绩很",
                      "因为他努力学习，但是成绩很", "虽然他努力学习，所以成绩很"],
        "targets" : ["优秀", "糟糕"]
    },
    "topic2" : {
        "contexts" : ["因为经济不景气，所以生意很", "虽然经济不景气，但是生意很",
                      "因为经济不景气，但是生意很", "虽然经济不景气，所以生意很"],
        "targets" : ["冷清", "火热"]
    },
    "topic3" : {
        "contexts" : ["因为工作没完成，所以情绪很", "虽然工作没完成，但是情绪很",
                      "因为工作没完成，但是情绪很", "虽然工作没完成，所以情绪很"],
        "targets" : ["焦急", "轻松"]
    },
    "topic4" : {
        "contexts" : ["因为加班了很久，所以精神很", "虽然加班了很久，但是精神很",
                      "因为加班了很久，但是精神很", "虽然加班了很久，所以精神很"],
        "targets" : ["萎靡", "饱满"]
    },
    "topic5" : {
        "contexts" : ["因为他经验不足，所以工作很", "虽然他经验不足，但是工作很",
                      "因为他经验不足，但是工作很", "虽然他经验不足，所以工作很"],
        "targets" : ["低效", "高效"]
    },
    "topic6" : {
        "contexts" : ["因为他遭遇挫折，所以意志很", "虽然他遭遇挫折，但是意志很",
                      "因为他遭遇挫折，但是意志很", "虽然他遭遇挫折，所以意志很"],
        "targets" : ["消沉", "高昂"]
    },
    "topic7" : {
        "contexts" : ["因为他年龄很大，所以动作很", "虽然他年龄很大，但是动作很",
                      "因为他年龄很大，但是动作很", "虽然他年龄很大，所以动作很"],
        "targets" : ["迟缓", "迅速"]
    }
}

BLOCK_MAPPING = {
    "block-1/sent-1" : "Sta1",
    "block-1/sent-2" : "B1Dev1",
    "block-1/sent-3" : "B1Dev2",
    "block-1/sent-4" : "Sta2",
    
    "block-2/sent-1" : "Sta1",
    "block-2/sent-2" : "B2Dev1",
    "block-2/sent-3" : "B2Dev2",
    "block-2/sent-4" : "Sta2",
    
    "block-3/sent-1" : "Sta1",
    "block-3/sent-2" : "B3Dev1",
    "block-3/sent-3" : "B3Dev2",
    "block-3/sent-4" : "Sta2",
}