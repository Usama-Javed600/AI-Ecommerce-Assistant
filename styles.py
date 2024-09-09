# CSS styling for the chat interface
css = '''
<style>

.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}

</style>
'''

# HTML template for the bot messages
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://t4.ftcdn.net/jpg/02/23/38/61/360_F_223386120_OMJd0gW045lI3TGy4UfUDOzOKvcrDWLR.jpg" style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# HTML template for the user messages
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGMNgCIQHLiS2yK7hDBg-1iljlOrMBErsaUA&s"style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
