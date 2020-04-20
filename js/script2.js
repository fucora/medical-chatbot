(function() {

    var chat = {
        messageToSend: '',
        messageResponses: [
            'Why did the web developer leave the restaurant? Because of the table layout.',
            'How do you comfort a JavaScript bug? You console it.',
            'An SQL query enters a bar, approaches two tables and asks: "May I join you?"',
            'What is the most used language in programming? Profanity.',
            'What is the object-oriented way to become wealthy? Inheritance.',
            'An SEO expert walks into a bar, bars, pub, tavern, public house, Irish pub, drinks, beer, alcohol'
        ],
        init: function() {
            this.cacheDOM();
            this.bindEvents();
            this.render();
        },
        cacheDOM: function() {
            this.$chatHistory = $('.chat-history');
            this.$button = $('button');
            this.$textarea = $('#message-to-send');
            this.$chatHistoryList = this.$chatHistory.find('ul');
        },
        bindEvents: function() {
            this.$button.on('click', this.addMessage.bind(this));
            this.$textarea.on('keyup', this.addMessageEnter.bind(this));
        },
        render: function(msg = '') {
            var returnCode = ''
            this.scrollToBottom();
            if (this.messageToSend.trim() !== '') {

                var template = Handlebars.compile($("#message-template").html());
                var context = {
                    question: this.messageToSend
                  
                };
                let that = this
                // console.log(this.messageToSend)
                // $.ajax({
                //     url: 'http://localhost:5000/api',
                //     method: "post",
                //     dataType: 'json',
                //     type:"post",
                //     data: context,
                //     headers:{
                //         "Access-Control-Allow-Origin":true,
                //     },
                //     success: function(msg) {
                //         returnCode = msg;
                //         console.log(returnCode.answer)
                //         that.render(returnCode.answer);
                //         // return returnCode.answer
                //         // if (returnCode == '200') {
				// 		// 	return returnCode
							
                //         // } else {
				// 		// 	console.log('失败')
                //         // }
                //     },
                //     error: function(e){
                //         console.log(e)
                //     }
                // });
   


                this.$chatHistoryList.append(template(context));
                this.scrollToBottom();
                this.$textarea.val('');

                // responses
                var templateResponse = Handlebars.compile($("#message-response-template").html());
                
                var contextResponse = {
                    // response: this.getRandomItem(this.messageResponses),//
                    response:msg,
                    time: this.getCurrentTime()
                };
                console.log(contextResponse)
                setTimeout(function() {
                    this.$chatHistoryList.append(templateResponse(contextResponse));
                    this.scrollToBottom();
                }.bind(this), 1500);

            }

        },

        addMessage: function() {
            console.log('666')
            
            let msg = this.$textarea.val()
            let that = this
            $.ajax({
                url: 'http://localhost:5000/api',
                method: "post",
                dataType: 'json',
                type:"post",
                data: {question:msg},
                headers:{
                    "Access-Control-Allow-Origin":true,
                },
                success: function(msg) {
                    returnCode = msg;
                    console.log(returnCode.answer)
                    that.messageToSend = that.$textarea.val()
                    that.render(returnCode.answer);
                    
                    // return returnCode.answer
                    // if (returnCode == '200') {
                    // 	return returnCode
                        
                    // } else {
                    // 	console.log('失败')
                    // }
                },
                error: function(e){
                    console.log(e)
                }
            });
            
            // this.render();
        },
        addMessageEnter: function(event) {
            // enter was pressed
            if (event.keyCode === 13) {
                this.addMessage();
            }
        },
        scrollToBottom: function() {
            this.$chatHistory.scrollTop(this.$chatHistory[0].scrollHeight);
        },
        getCurrentTime: function() {
            return new Date().toLocaleTimeString().
            replace(/([\d]+:[\d]{2})(:[\d]{2})(.*)/, "$1$3");
        },
        getRandomItem: function(arr) {
            return arr[Math.floor(Math.random() * arr.length)];
        }

    };

    chat.init();

    var searchFilter = {
        options: { valueNames: ['name'] },
        init: function() {
            var userList = new List('people-list', this.options);
            var noItems = $('<li id="no-items-found">No items found</li>');

            userList.on('updated', function(list) {
                if (list.matchingItems.length === 0) {
                    $(list.list).append(noItems);
                } else {
                    noItems.detach();
                }
            });
        }
    };

    searchFilter.init();

})();

// function send() {
//     var sendtxt = $("#message-to-send").val();
//     console.log(sendtxt);
// }