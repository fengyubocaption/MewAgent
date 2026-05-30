const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            abortController: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            showHistorySidebar: false,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            selectedFile: null,
            isUploading: false,
            uploadProgress: '',
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false,
            // Resume
            resumes: [],
            resumesLoading: false,
            selectedResumeFile: null,
            isUploadingResume: false,
            resumeUploadProgress: '',
            resumeDetail: null,
            // JD
            jds: [],
            jdsLoading: false,
            jdForm: { title: '', company: '', jd_text: '' },
            isCreatingJD: false,
            jdCreateProgress: '',
            jdDetail: null,
            matchResumeId: '',
            isMatching: false,
            matchResult: '',
            // Mock Interview
            mockInterview: {
                jdId: '',
                type: '综合面',
                loading: false,
                questions: '',
                answer: '',
                questionText: '',
                evaluating: false,
                evaluation: ''
            }
        };
    },
    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },
        isAdmin() {
            return this.currentUser?.role === 'admin';
        }
    },
    async mounted() {
        this.configureMarked();
        if (this.token) {
            try {
                await this.fetchMe();
            } catch (_) {
                this.handleLogout();
            }
        }
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(url, opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error('登录已过期，请重新登录');
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/auth/me');
            if (!response.ok) {
                throw new Error('认证失败');
            }
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert('用户名和密码不能为空');
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === 'login' ? '/auth/login' : '/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '认证失败');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.sessionId = 'session_' + Date.now();
                this.activeNav = 'newChat';
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.resumes = [];
            this.jds = [];
            this.resumeDetail = null;
            this.jdDetail = null;
            this.matchResult = '';
            this.mockInterview.questions = '';
            this.mockInterview.evaluation = '';
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
            localStorage.removeItem('accessToken');
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },

        async handleSend() {
            if (!this.isAuthenticated) {
                alert('请先登录');
                return;
            }

            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;

            this.messages.push({
                text: text,
                isUser: true
            });

            this.userInput = '';
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            this.messages.push({
                text: '',
                isUser: false,
                isThinking: true,
                ragTrace: null,
                ragSteps: []
            });
            const botMsgIdx = this.messages.length - 1;

            this.abortController = new AbortController();

            try {
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        session_id: this.sessionId
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') {
                                    if (this.messages[botMsgIdx].isThinking) {
                                        this.messages[botMsgIdx].isThinking = false;
                                    }
                                    this.messages[botMsgIdx].text += data.content;
                                } else if (data.type === 'trace') {
                                    this.messages[botMsgIdx].ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    if (!this.messages[botMsgIdx].ragSteps) {
                                        this.messages[botMsgIdx].ragSteps = [];
                                    }
                                    this.messages[botMsgIdx].ragSteps.push(data.step);
                                } else if (data.type === 'error') {
                                    this.messages[botMsgIdx].isThinking = false;
                                    this.messages[botMsgIdx].text += `\n[Error: ${data.content}]`;
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    this.messages[botMsgIdx].isThinking = false;
                    if (!this.messages[botMsgIdx].text) {
                        this.messages[botMsgIdx].text = '(已终止回答)';
                    } else {
                        this.messages[botMsgIdx].text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    this.messages[botMsgIdx].isThinking = false;
                    this.messages[botMsgIdx].text = `喵呜... 出了点问题：${error.message}`;
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
        },

        handleClearChat() {
            if (confirm('确定要清空当前对话吗？喵？')) {
                this.messages = [];
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'history';
            this.showHistorySidebar = true;
            try {
                const response = await this.authFetch('/sessions');
                if (!response.ok) {
                    throw new Error('Failed to load sessions');
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                alert('加载历史记录失败：' + error.message);
            }
        },

        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.showHistorySidebar = false;
            this.activeNav = 'newChat';

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null
                }));

                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                alert('加载会话失败：' + error.message);
                this.messages = [];
            }
        },

        async deleteSession(sessionId) {
            if (!confirm(`确定要删除会话 "${sessionId}" 吗？`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeNav = 'newChat';
                }

                if (payload.message) {
                    alert(payload.message);
                }
            } catch (error) {
                alert('删除会话失败：' + error.message);
            }
        },

        handleSettings() {
            if (!this.isAdmin) {
                alert('仅管理员可访问文档管理');
                return;
            }
            this.activeNav = 'settings';
            this.showHistorySidebar = false;
            this.loadDocuments();
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await this.authFetch('/documents');
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || 'Failed to load documents');
                }
                const data = await response.json();
                this.documents = data.documents;
            } catch (error) {
                alert('加载文档列表失败：' + error.message);
            } finally {
                this.documentsLoading = false;
            }
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedFile = files[0];
                this.uploadProgress = '';
            }
        },

        async uploadDocument() {
            if (!this.selectedFile) {
                alert('请先选择文件');
                return;
            }

            this.isUploading = true;
            this.uploadProgress = '正在上传...';

            try {
                const formData = new FormData();
                formData.append('file', this.selectedFile);

                const response = await this.authFetch('/documents/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Upload failed');
                }

                const data = await response.json();
                this.uploadProgress = data.message;

                this.selectedFile = null;
                if (this.$refs.fileInput) {
                    this.$refs.fileInput.value = '';
                }

                await this.loadDocuments();

                setTimeout(() => {
                    this.uploadProgress = '';
                }, 3000);

            } catch (error) {
                this.uploadProgress = '上传失败：' + error.message;
            } finally {
                this.isUploading = false;
            }
        },

        async deleteDocument(filename) {
            if (!confirm(`确定要删除文档 "${filename}" 吗？这将同时删除 Milvus 中的所有相关向量。`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/documents/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Delete failed');
                }

                const data = await response.json();
                alert(data.message);
                await this.loadDocuments();

            } catch (error) {
                alert('删除文档失败：' + error.message);
            }
        },

        getFileIcon(fileType) {
            if (fileType === 'PDF') {
                return 'fas fa-file-pdf';
            } else if (fileType === 'Word') {
                return 'fas fa-file-word';
            } else if (fileType === 'Excel') {
                return 'fas fa-file-excel';
            }
            return 'fas fa-file';
        },

        // ==================== Resume ====================
        async handleResume() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'resume';
            this.showHistorySidebar = false;
            this.resumeDetail = null;
            await this.loadResumes();
        },

        async loadResumes() {
            this.resumesLoading = true;
            try {
                const response = await this.authFetch('/resume');
                if (!response.ok) throw new Error('加载失败');
                const data = await response.json();
                this.resumes = data.resumes;
            } catch (error) {
                alert('加载简历列表失败：' + error.message);
            } finally {
                this.resumesLoading = false;
            }
        },

        handleResumeFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedResumeFile = files[0];
                this.resumeUploadProgress = '';
            }
        },

        async uploadResume() {
            if (!this.selectedResumeFile) return;
            this.isUploadingResume = true;
            this.resumeUploadProgress = '正在上传并解析...';
            try {
                const formData = new FormData();
                formData.append('file', this.selectedResumeFile);
                const response = await this.authFetch('/resume/upload', { method: 'POST', body: formData });
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || '上传失败');
                }
                const data = await response.json();
                this.resumeUploadProgress = data.message;
                this.selectedResumeFile = null;
                if (this.$refs.resumeFileInput) this.$refs.resumeFileInput.value = '';
                await this.loadResumes();
                setTimeout(() => { this.resumeUploadProgress = ''; }, 3000);
            } catch (error) {
                this.resumeUploadProgress = '上传失败：' + error.message;
            } finally {
                this.isUploadingResume = false;
            }
        },

        async viewResumeDetail(id) {
            try {
                const response = await this.authFetch(`/resume/${id}`);
                if (!response.ok) throw new Error('加载失败');
                this.resumeDetail = await response.json();
            } catch (error) {
                alert('加载简历详情失败：' + error.message);
            }
        },

        async deleteResume(id) {
            if (!confirm('确定要删除这份简历吗？')) return;
            try {
                const response = await this.authFetch(`/resume/${id}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('删除失败');
                this.resumes = this.resumes.filter(r => r.id !== id);
                if (this.resumeDetail?.id === id) this.resumeDetail = null;
            } catch (error) {
                alert('删除简历失败：' + error.message);
            }
        },

        // ==================== JD ====================
        async handleJD() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'jd';
            this.showHistorySidebar = false;
            this.jdDetail = null;
            this.matchResult = '';
            await this.loadJDs();
            await this.loadResumes(); // for match selector
        },

        async loadJDs() {
            this.jdsLoading = true;
            try {
                const response = await this.authFetch('/jd');
                if (!response.ok) throw new Error('加载失败');
                const data = await response.json();
                this.jds = data.job_descriptions;
            } catch (error) {
                alert('加载 JD 列表失败：' + error.message);
            } finally {
                this.jdsLoading = false;
            }
        },

        async createJD() {
            const jdText = this.jdForm.jd_text.trim();
            if (!jdText || jdText.length < 20) {
                alert('请输入完整的职位描述（至少20字）');
                return;
            }
            this.isCreatingJD = true;
            this.jdCreateProgress = '正在分析...';
            try {
                const response = await this.authFetch('/jd', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.jdForm)
                });
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || '创建失败');
                }
                const data = await response.json();
                this.jdCreateProgress = data.message;
                this.jdForm = { title: '', company: '', jd_text: '' };
                await this.loadJDs();
                setTimeout(() => { this.jdCreateProgress = ''; }, 3000);
            } catch (error) {
                this.jdCreateProgress = '创建失败：' + error.message;
            } finally {
                this.isCreatingJD = false;
            }
        },

        async viewJDDetail(id) {
            try {
                const response = await this.authFetch(`/jd/${id}`);
                if (!response.ok) throw new Error('加载失败');
                this.jdDetail = await response.json();
                this.matchResult = '';
                this.matchResumeId = '';
            } catch (error) {
                alert('加载 JD 详情失败：' + error.message);
            }
        },

        async deleteJD(id) {
            if (!confirm('确定要删除这个 JD 吗？')) return;
            try {
                const response = await this.authFetch(`/jd/${id}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('删除失败');
                this.jds = this.jds.filter(j => j.id !== id);
                if (this.jdDetail?.id === id) this.jdDetail = null;
            } catch (error) {
                alert('删除 JD 失败：' + error.message);
            }
        },

        async runMatch() {
            if (!this.matchResumeId || !this.jdDetail) return;
            this.isMatching = true;
            this.matchResult = '';
            try {
                const msg = `请分析简历(ID:${this.matchResumeId})与JD(ID:${this.jdDetail.id})的匹配度`;
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, session_id: this.sessionId })
                });
                if (!response.ok) throw new Error('匹配分析失败');
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);
                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') this.matchResult += data.content;
                            } catch (e) {}
                        }
                    }
                }
            } catch (error) {
                this.matchResult = '匹配分析失败：' + error.message;
            } finally {
                this.isMatching = false;
            }
        },

        // ==================== Mock Interview ====================
        async handleMockInterview() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'mockInterview';
            this.showHistorySidebar = false;
            if (this.jds.length === 0) await this.loadJDs();
        },

        async startMockInterview() {
            if (!this.mockInterview.jdId) return;
            this.mockInterview.loading = true;
            this.mockInterview.questions = '';
            this.mockInterview.evaluation = '';
            this.mockInterview.answer = '';
            this.mockInterview.questionText = '';
            try {
                const msg = `请根据JD(ID:${this.mockInterview.jdId})生成${this.mockInterview.type}面试题`;
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, session_id: this.sessionId })
                });
                if (!response.ok) throw new Error('生成面试题失败');
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);
                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') this.mockInterview.questions += data.content;
                            } catch (e) {}
                        }
                    }
                }
            } catch (error) {
                this.mockInterview.questions = '生成失败：' + error.message;
            } finally {
                this.mockInterview.loading = false;
            }
        },

        async evaluateAnswer() {
            if (!this.mockInterview.answer || !this.mockInterview.questionText) return;
            this.mockInterview.evaluating = true;
            this.mockInterview.evaluation = '';
            try {
                const msg = `请评估我对这道面试题的回答。\n\n题目：${this.mockInterview.questionText}\n\n我的回答：${this.mockInterview.answer}`;
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, session_id: this.sessionId })
                });
                if (!response.ok) throw new Error('评估失败');
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);
                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') this.mockInterview.evaluation += data.content;
                            } catch (e) {}
                        }
                    }
                }
            } catch (error) {
                this.mockInterview.evaluation = '评估失败：' + error.message;
            } finally {
                this.mockInterview.evaluating = false;
            }
        }
    },
    watch: {
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            },
            deep: true
        }
    }
}).mount('#app');
