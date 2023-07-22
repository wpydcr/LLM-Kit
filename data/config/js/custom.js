var ga = document.getElementsByTagName("gradio-app");
var targetNode = ga[0];
function gradioLoaded(mutations) {
    for (var i = 0; i < mutations.length; i++) {
        if (mutations[i].addedNodes.length) {
            setTips();
            live2d();
            keyboard_enter();
            
        }
    }

}


// 监视页面内部 DOM 变动
var observer = new MutationObserver(function (mutations) {
    gradioLoaded(mutations);
});
observer.observe(targetNode, { childList: true, subtree: true });


function setTipsByID(element, text) {
    element.setAttribute('title', text);
}

function setTips() {
    chat_api_list = document.querySelectorAll('#chat_api_list label');
    chat_ernie_temperature = document.querySelector('#chat_ernie_temperature');
    chat_ernie_top_p = document.querySelector('#chat_ernie_top_p');
    chat_ernie_penalty_score = document.querySelector('#chat_ernie_penalty_score');
    chat_chatglm_temperature = document.querySelector('#chat_chatglm_temperature');
    chat_chatglm_top_p = document.querySelector('#chat_chatglm_top_p');
    chat_spark_temperature = document.querySelector('#chat_spark_temperature');
    chat_spark_top_k = document.querySelector('#chat_spark_top_k');
    chat_ali_top_p = document.querySelector('#chat_ali_top_p');
    chat_ali_top_k = document.querySelector('#chat_ali_top_k');
    chat_model_max_length = document.querySelector('#chat_model_max_length');
    chat_model_top_p = document.querySelector('#chat_model_top_p');
    chat_model_temperature = document.querySelector('#chat_model_temperature');
    if (chat_api_list.length > 0) {
        setTipsByID(chat_api_list[0], '需要科学上网才能使用哦~');
    }
    if (chat_ernie_temperature) {
        setTipsByID(chat_ernie_temperature, '较高的数值会使输出更加随机，而较低的数值会使其更加集中和确定');
    }
    if (chat_ernie_top_p) {
        setTipsByID(chat_ernie_top_p, '影响输出文本的多样性，取值越大，生成文本的多样性越强');
    }
    if (chat_ernie_penalty_score) {
        setTipsByID(chat_ernie_penalty_score, '通过对已生成的token增加惩罚，减少重复生成的现象');
    }
    if (chat_chatglm_temperature) {
        setTipsByID(chat_chatglm_temperature, '较高的数值会使输出更加随机，而较低的数值会使其更加集中和确定');
    }
    if (chat_chatglm_top_p) {
        setTipsByID(chat_chatglm_top_p, '影响输出文本的多样性，取值越大，生成文本的多样性越强');
    }
    if (chat_spark_temperature) {
        setTipsByID(chat_spark_temperature, '较高的数值会使输出更加随机，而较低的数值会使其更加集中和确定');
    }
    if (chat_spark_top_k) {
        setTipsByID(chat_spark_top_k, '影响输出文本的多样性，取值越大，生成文本的多样性越强');
    }
    if (chat_ali_top_p) {
        setTipsByID(chat_ali_top_p, '影响输出文本的多样性，取值越大，生成文本的多样性越强');
    }
    if (chat_ali_top_k) {
        setTipsByID(chat_ali_top_k, '影响输出文本的多样性，取值越大，生成文本的多样性越强');
    }
    if (chat_model_max_length) {
        setTipsByID(chat_model_max_length, '生成文本的最大长度');
    }
    if (chat_model_top_p) {
        setTipsByID(chat_model_top_p, '影响输出文本的多样性，取值越大，生成文本的多样性越强');
    }
    if (chat_model_temperature) {
        setTipsByID(chat_model_temperature, '较高的数值会使输出更加随机，而较低的数值会使其更加集中和确定');
    }
}

function generateUID() {
    var timestamp = new Date().getTime().toString(36);
    var randomNum = Math.random().toString(36).substr(2, 5);
    var uid = timestamp + randomNum;
    return uid;
}

function live2dShow() {
    var uid = generateUID();

    var live2d_div = document.querySelector("#live2d-div");
    var live2d_canvas = document.querySelector("#live2d-canvas");

    if (live2d_canvas != null) {
        var script1 = document.querySelector(`script[src^="file=data/config/js/live2dcubismcore.min.js"]`);
        var script2 = document.querySelector(`script[src^="file=data/config/js/live2d.min.js"]`);
        var script3 = document.querySelector(`script[src^="file=data/config/js/pixi.min.js"]`);
        var script4 = document.querySelector(`script[src^="file=data/config/js/index.js"]`);
        document.body.removeChild(script1);
        document.body.removeChild(script2);
        document.body.removeChild(script3);
        document.body.removeChild(script4);
    }


    if (live2d_canvas == null) {
        live2d_div.innerHTML = "";
        var canvas = document.createElement("canvas");
        canvas.id = "live2d-canvas";
        live2d_div.appendChild(canvas);
    }


    var script = document.createElement("script");
    script.src = "file=data/config/js/live2dcubismcore.min.js";
    document.body.appendChild(script);
    var script = document.createElement("script");
    script.src = "file=data/config/js/live2d.min.js";
    document.body.appendChild(script);
    var script = document.createElement("script");
    script.src = "file=data/config/js/pixi.min.js";
    document.body.appendChild(script);
    var script = document.createElement("script");
    script.type = "module"; script.crossOrigin = "anonymous"; script.src = "file=data/config/js/index.js?" + uid;
    document.body.appendChild(script);
}

function live2d() {
    var show_type_labels = document.querySelectorAll('#show_type label');
    if (show_type_labels.length > 0) {
        if (show_type_labels[0].getAttribute('data-listener') == null) {
            show_type_labels[0].setAttribute('data-listener', 'true');
            show_type_labels[0].addEventListener('change', function () {
                live2dShow();
            });
        }
    }

    var player_page = document.querySelectorAll('button');
    for (var i = 0; i < player_page.length; i++) {
        if (player_page[i].innerHTML.includes('角色扮演')) {
            if (player_page[i].getAttribute('data-listener') == null) {
                player_page[i].setAttribute('data-listener', 'true');
                player_page[i].addEventListener('click', function () {
                    if (show_type_labels.length > 0 && show_type_labels[0].className.indexOf('selected') > -1) {
                        live2dShow();
                    }
                });
            }
            break;
        }
    }
}

function keyboard_enter() {
    keyboard_enter_page('#chat-submitGroup', '#chat-submitBtn', '#chat-user-input textarea');
    keyboard_enter_page('#player-submitGroup', '#player-submitBtn', '#player-user-input textarea');
    keyboard_enter_page('#faiss-submitGroup', '#faiss-submitBtn', '#faiss-user-input textarea');
    keyboard_enter_page('#mysql-submitGroup', '#mysql-submitBtn', '#mysql-user-input textarea');
}

function keyboard_enter_page(submitGroup_id, submitBtn_id, user_input_id) {

    var submitGroup = document.querySelector(submitGroup_id);
    var submitBtn = document.querySelector(submitBtn_id);
    var user_input = document.querySelector(user_input_id);
    if(submitGroup){
        if(submitGroup.getAttribute('data-listener') == null){
            submitGroup.setAttribute('data-listener', 'true');
            user_input.addEventListener('keydown', function (e) {
                if (e.ctrlKey && e.code === 'Enter') {
                    user_input.value += '\n';
                }else if (!e.ctrlKey && e.code === 'Enter') {
                    e.preventDefault();
                    if(submitGroup.className.indexOf('hidden') == -1){
                        submitBtn.click();
                    }
                }
            });
        }
    }
}